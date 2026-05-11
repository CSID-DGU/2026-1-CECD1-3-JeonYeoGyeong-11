"""Dominance-aware FedAvgM variants for Phase-3 diagnostics."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from spectral_fl.projection import flatten_weights
from spectral_fl.strategies.baselines.ordering import sort_fit_results_by_cid
from spectral_fl.strategies.baselines.tracing import _EvalTracer, _InteractionDiagnostics
from spectral_fl.strategies.spectral.momentum import apply_server_optimizer


def _safe_cosine(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    xn = float(np.linalg.norm(x))
    yn = float(np.linalg.norm(y))
    if xn <= eps or yn <= eps:
        return 0.0
    return float(np.dot(x, y) / (xn * yn + eps))


class TracingDominanceAwareFedAvgM(
    _InteractionDiagnostics, _EvalTracer, fl.server.strategy.FedAvg
):
    """FedAvgM with optional dominance-oriented correction before server step."""

    def __init__(
        self,
        *,
        dominance_mode: str = "fedavgm",
        dominance_tau: float = 1.0,
        dominance_threshold: float = 0.35,
        clip_norm: float = 0.0,
        clip_percentile: float = 0.75,
        contribution_cap: float = 0.0,
        contribution_cap_percentile: float = 0.75,
        contribution_cap_kappa: float = 0.0,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.9,
        **kwargs,
    ) -> None:
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)
        _InteractionDiagnostics.__init__(self)
        self.dominance_mode = str(dominance_mode).strip().lower()
        self.dominance_tau = float(dominance_tau)
        self.dominance_threshold = float(dominance_threshold)
        self.clip_norm = float(clip_norm)
        self.clip_percentile = float(clip_percentile)
        self.contribution_cap = float(contribution_cap)
        self.contribution_cap_percentile = float(contribution_cap_percentile)
        self.contribution_cap_kappa = float(contribution_cap_kappa)
        self.server_learning_rate = float(server_learning_rate)
        self.server_momentum = float(server_momentum)
        self.server_momentum_vector: Optional[NDArrays] = None
        self._prev_dominant_cid: Optional[str] = None
        self._eps = 1e-12

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self._cache_current_global(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)

    @staticmethod
    def _unflatten_like(vec: np.ndarray, ref: NDArrays) -> NDArrays:
        out: NDArrays = []
        cursor = 0
        for arr in ref:
            size = int(arr.size)
            piece = vec[cursor : cursor + size].reshape(arr.shape)
            out.append(piece.astype(arr.dtype, copy=False))
            cursor += size
        return out

    def _effective_q_stats(self, q_vals: np.ndarray) -> Tuple[np.ndarray, float, float]:
        q_bar = q_vals / (float(np.sum(q_vals)) + self._eps)
        dominance_ratio = float(np.max(q_bar))
        n_eff = float(1.0 / (np.sum(np.square(q_bar)) + self._eps))
        return q_bar, dominance_ratio, n_eff

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}
        if self._current_global is None:
            return super().aggregate_fit(server_round, results, failures)

        ordered = sort_fit_results_by_cid(list(results))
        cids: List[str] = []
        local_weights: List[NDArrays] = []
        n_examples: List[int] = []
        for proxy, fit_res in ordered:
            cids.append(self._safe_client_id(proxy, fit_res))
            local_weights.append(parameters_to_ndarrays(fit_res.parameters))
            n_examples.append(int(fit_res.num_examples))

        local_updates: List[NDArrays] = [
            [lp - gp for lp, gp in zip(local, self._current_global)]
            for local in local_weights
        ]
        flat_updates = np.stack(
            [flatten_weights(delta).astype(np.float64, copy=False) for delta in local_updates],
            axis=0,
        )
        update_norms = np.asarray(
            [float(np.linalg.norm(vec)) for vec in flat_updates], dtype=np.float64
        )
        n_examples_arr = np.asarray(n_examples, dtype=np.float64)
        p = n_examples_arr / max(float(np.sum(n_examples_arr)), self._eps)

        q_raw = p * update_norms
        q_bar_raw, di_raw, n_eff_raw = self._effective_q_stats(q_raw)
        dominant_idx = int(np.argmax(q_bar_raw))
        dominant_cid = cids[dominant_idx]
        dominant_vec = flat_updates[dominant_idx]

        corrected_flat = flat_updates.copy()
        alpha = p.copy()
        trigger_applied = False
        clip_value = float(self.clip_norm)
        cap_value = float(self.contribution_cap)

        if self.dominance_mode == "uniform":
            alpha = np.ones_like(p) / float(len(p))
        elif self.dominance_mode == "norm_clip":
            if clip_value <= 0.0:
                pct = min(max(self.clip_percentile, 0.0), 1.0)
                clip_value = float(np.quantile(update_norms, pct))
            scales = np.minimum(1.0, clip_value / (update_norms + self._eps))
            corrected_flat = corrected_flat * scales[:, None]
        elif self.dominance_mode == "contribution_cap":
            if cap_value <= 0.0:
                if self.contribution_cap_kappa > 0.0:
                    cap_value = float(self.contribution_cap_kappa * np.median(q_raw))
                else:
                    pct = min(max(self.contribution_cap_percentile, 0.0), 1.0)
                    cap_value = float(np.quantile(q_raw, pct))
            scales = np.minimum(1.0, cap_value / (q_raw + self._eps))
            corrected_flat = corrected_flat * scales[:, None]
        elif self.dominance_mode == "soft_reweight":
            raw_alpha = p * np.exp(-float(self.dominance_tau) * q_bar_raw)
            alpha = raw_alpha / (float(np.sum(raw_alpha)) + self._eps)
        elif self.dominance_mode == "n_eff_aware":
            # Boost low-contribution clients to increase effective participation.
            raw_alpha = p / np.sqrt(np.maximum(q_bar_raw, self._eps))
            alpha = raw_alpha / (float(np.sum(raw_alpha)) + self._eps)
        elif self.dominance_mode == "triggered_soft_reweight":
            trigger_applied = bool(di_raw > float(self.dominance_threshold))
            if trigger_applied:
                raw_alpha = p * np.exp(-float(self.dominance_tau) * q_bar_raw)
                alpha = raw_alpha / (float(np.sum(raw_alpha)) + self._eps)
        elif self.dominance_mode == "fedavgm":
            pass
        else:
            raise ValueError(f"Unknown dominance_mode: {self.dominance_mode!r}")

        raw_delta = np.sum(p[:, None] * flat_updates, axis=0)
        corrected_delta = np.sum(alpha[:, None] * corrected_flat, axis=0)

        candidate_global = [
            gp + delta
            for gp, delta in zip(
                self._current_global,
                self._unflatten_like(corrected_delta, local_updates[0]),
            )
        ]
        new_global, self.server_momentum_vector, server_diag = apply_server_optimizer(
            current_global=self._current_global,
            candidate_global=candidate_global,
            server_learning_rate=self.server_learning_rate,
            server_momentum=self.server_momentum,
            server_momentum_vector=self.server_momentum_vector,
        )

        corrected_norms = np.asarray(
            [float(np.linalg.norm(vec)) for vec in corrected_flat], dtype=np.float64
        )
        q_corrected = alpha * corrected_norms
        q_bar_corrected, di_corrected, n_eff_corrected = self._effective_q_stats(q_corrected)
        top3_mass = float(np.sum(np.sort(q_bar_corrected)[-3:])) if q_bar_corrected.size > 0 else float("nan")

        dominant_changed = (
            None
            if self._prev_dominant_cid is None
            else bool(self._prev_dominant_cid != dominant_cid)
        )
        self._prev_dominant_cid = dominant_cid

        self._record_interaction_diagnostics(
            server_round=server_round,
            results=list(results),
            method=f"dominance_{self.dominance_mode}",
        )
        base_log = self.round_logs[-1]
        base_log.update(
            {
                "method": f"dominance_{self.dominance_mode}",
                "dominance_mode": self.dominance_mode,
                "dominance_tau": float(self.dominance_tau),
                "dominance_threshold": float(self.dominance_threshold),
                "dominance_triggered": bool(trigger_applied),
                "update_definition": "g_i = w_i_local - w_t",
                "max_client_weight": float(np.max(p)),
                "dominant_client_index": int(dominant_idx),
                "dominant_client_id": str(dominant_cid),
                "dominant_client_changed": dominant_changed,
                "dominance_ratio_raw": float(di_raw),
                "effective_num_clients_raw": float(n_eff_raw),
                "dominance_ratio_corrected": float(di_corrected),
                "effective_num_clients_corrected": float(n_eff_corrected),
                "client_contribution_normalized_raw": [float(x) for x in q_bar_raw.tolist()],
                "client_contribution_normalized_corrected": [
                    float(x) for x in q_bar_corrected.tolist()
                ],
                "client_aggregation_weights_alpha": [float(x) for x in alpha.tolist()],
                "fedavg_delta_norm": float(np.linalg.norm(raw_delta)),
                "base_delta_norm": float(np.linalg.norm(raw_delta)),
                "corrected_delta_norm": float(np.linalg.norm(corrected_delta)),
                "delta_corrected_over_fedavg_norm": float(
                    np.linalg.norm(corrected_delta) / (np.linalg.norm(raw_delta) + self._eps)
                ),
                "cosine_raw_vs_corrected_delta": _safe_cosine(raw_delta, corrected_delta),
                "cos_delta_corrected_vs_base": _safe_cosine(raw_delta, corrected_delta),
                "relative_delta_change": float(
                    np.linalg.norm(corrected_delta - raw_delta)
                    / (np.linalg.norm(raw_delta) + self._eps)
                ),
                "rel_delta_change": float(
                    np.linalg.norm(corrected_delta - raw_delta)
                    / (np.linalg.norm(raw_delta) + self._eps)
                ),
                "cosine_dominant_vs_raw_delta": _safe_cosine(dominant_vec, raw_delta),
                "cosine_dominant_vs_corrected_delta": _safe_cosine(
                    dominant_vec, corrected_delta
                ),
                "client_contribution_weights": [float(x) for x in alpha.tolist()],
                "max_contribution": float(np.max(q_bar_corrected)),
                "top1_client_id_by_contribution": str(dominant_cid),
                "top3_contribution_mass": top3_mass,
                "contribution_entropy": float(
                    -np.sum(q_bar_corrected * np.log(np.maximum(q_bar_corrected, self._eps)))
                ),
                "clip_norm": float(clip_value),
                "contribution_cap": float(cap_value),
                "contribution_cap_kappa": float(self.contribution_cap_kappa),
                "server_optimizer": str(server_diag.get("server_optimizer", "")),
                "server_candidate_delta_norm": float(
                    server_diag.get("server_candidate_delta_norm", np.nan)
                ),
                "server_applied_delta_norm": float(
                    server_diag.get("server_applied_delta_norm", np.nan)
                ),
            }
        )
        metrics = {
            "DI_t_raw": float(di_raw),
            "N_eff_t_raw": float(n_eff_raw),
            "DI_t_corrected": float(di_corrected),
            "N_eff_t_corrected": float(n_eff_corrected),
            "delta_norm_raw": float(np.linalg.norm(raw_delta)),
            "delta_norm_corrected": float(np.linalg.norm(corrected_delta)),
            "delta_cosine_raw_corrected": _safe_cosine(raw_delta, corrected_delta),
            "cosine_dom_raw": _safe_cosine(dominant_vec, raw_delta),
            "cosine_dom_corrected": _safe_cosine(dominant_vec, corrected_delta),
        }
        return ndarrays_to_parameters(new_global), metrics


__all__ = ["TracingDominanceAwareFedAvgM"]
