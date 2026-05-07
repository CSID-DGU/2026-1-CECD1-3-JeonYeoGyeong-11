"""Thin baseline strategy wrappers that record per-client evaluation logs."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import flwr as fl
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy


class _EvalTracer:
    """Mixin providing eval_logs[round] = [{cid, accuracy, loss, num_examples}]."""

    def __init__(self) -> None:
        self.eval_logs: Dict[int, List[Dict[str, Any]]] = {}

    def _record_eval(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> None:
        per_client = []
        for proxy, eval_res in results:
            metrics = eval_res.metrics or {}
            per_client.append(
                {
                    "cid": str(getattr(proxy, "cid", "?")),
                    "accuracy": float(metrics.get("accuracy", float("nan"))),
                    "loss": float(eval_res.loss),
                    "num_examples": int(eval_res.num_examples),
                }
            )
        self.eval_logs[int(server_round)] = per_client


class TracingFedAvg(_EvalTracer, fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedAvgM(_EvalTracer, fl.server.strategy.FedAvgM):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAvgM.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedAdagrad(_EvalTracer, fl.server.strategy.FedAdagrad):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAdagrad.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedAdam(_EvalTracer, fl.server.strategy.FedAdam):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAdam.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedYogi(_EvalTracer, fl.server.strategy.FedYogi):
    def __init__(self, **kwargs):
        fl.server.strategy.FedYogi.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedMedian(_EvalTracer, fl.server.strategy.FedMedian):
    def __init__(self, **kwargs):
        fl.server.strategy.FedMedian.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedTrimmedAvg(_EvalTracer, fl.server.strategy.FedTrimmedAvg):
    def __init__(self, **kwargs):
        fl.server.strategy.FedTrimmedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class TracingFedProx(_EvalTracer, fl.server.strategy.FedProx):
    def __init__(self, **kwargs):
        fl.server.strategy.FedProx.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)
