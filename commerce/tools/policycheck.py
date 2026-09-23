# -*- coding: utf-8 -*-
"""경로 규칙으로 잡히지 않는 위험 신호 검사.

    python -m commerce.tools.gate policy

CODEOWNERS는 "어떤 파일을 고쳤는가"만 본다. 그런데 소유자가 자기 경로 안에서
FL를 켜거나 미구현 게이트를 통과한 것처럼 적는 변경은 경로만 보면 평범하다.
이 검사는 그런 상태를 직접 확인해 자동 병합을 멈춘다.

판정은 종료 코드 0과 마지막 줄
`POLICY OK: fl=<n> gates=<n> contracts=<n> failures=0`의 동시 충족이다.
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACTS_DOC = REPO_ROOT / "docs/contracts.md"
RUN_LOCAL = REPO_ROOT / "commerce/deploy/run_local.py"
SCHEMA_DIR = REPO_ROOT / "commerce/packages/contracts/schemas"

# 문서 표의 상태 문자열. '실행 가능 (업무 검증 제외)'처럼 뒤에 단서가 붙을 수 있다.
IMPLEMENTED_PREFIX = "실행 가능"
NOT_IMPLEMENTED = "미구현"


# --- 1. FL은 기본적으로 꺼져 있어야 한다 --------------------------------------

def _check_fl_defaults(fail) -> int:
    from commerce.packages.fl_client.lifecycle import FLClientConfig

    checked = 0
    default = FLClientConfig()
    checked += 1
    if default.enabled is not False:
        fail("commerce/packages/fl_client/lifecycle.py",
             "FLClientConfig 기본값이 enabled=True다. FL 전송은 아직 구현·검증되지 않았다")
    checked += 1
    if default.mode != "protected":
        fail("commerce/packages/fl_client/lifecycle.py",
             "FLClientConfig 기본 mode가 protected가 아니다: %r" % default.mode)

    # 런처가 판매자에게 넘기는 FL_ENABLED 기본값.
    tree = ast.parse(RUN_LOCAL.read_text(encoding="utf-8"))
    launcher_values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                        and key.value.startswith("FL_") and isinstance(value, ast.Constant)):
                    launcher_values[key.value] = value.value
    checked += 1
    if launcher_values.get("FL_ENABLED") != "false":
        fail("commerce/deploy/run_local.py",
             "런처가 FL_ENABLED=%r로 기동한다. 로컬 실행은 FL를 켜지 않는다"
             % launcher_values.get("FL_ENABLED"))
    return checked


# --- 2. 게이트 상태 주장이 실제 동작과 같아야 한다 ----------------------------

def _gate_status_from_doc() -> dict:
    status = {}
    for line in CONTRACTS_DOC.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        name = cells[0].split(" ")[0]
        if re.fullmatch(r"[a-z][a-z0-9]*", name):
            status[name] = cells[-1]
    return status


def _check_gate_honesty(fail) -> int:
    from commerce.tools.gate import GATES

    documented = _gate_status_from_doc()
    checked = 0
    for name, (_, runner) in GATES.items():
        claimed = documented.get(name)
        if claimed is None:
            continue  # docs 게이트가 표 누락을 따로 잡는다.
        checked += 1
        is_stub = getattr(runner, "not_implemented", False)
        if claimed.startswith(NOT_IMPLEMENTED) and not is_stub:
            fail("commerce/tools/gate.py",
                 "%s 는 문서에서 미구현인데 실제 검사가 연결돼 있다. 표를 갱신하라" % name)
        if claimed.startswith(IMPLEMENTED_PREFIX) and is_stub:
            fail(str(CONTRACTS_DOC.relative_to(REPO_ROOT)),
                 "%s 를 '실행 가능'이라고 적었지만 NOT_IMPLEMENTED를 반환한다" % name)
    return checked


# --- 3. 계약이 조용히 사라지지 않아야 한다 ------------------------------------

def _check_contract_set(fail) -> int:
    text = CONTRACTS_DOC.read_text(encoding="utf-8")
    listed = set(re.findall(r"\b([a-z_]+\.v\d+)\b", text))
    on_disk = {p.name[: -len(".schema.json")] for p in SCHEMA_DIR.glob("*.schema.json")}
    for name in sorted(listed - on_disk):
        fail(str(CONTRACTS_DOC.relative_to(REPO_ROOT)),
             "문서에 있는 계약의 schema 파일이 없다: %s" % name)
    for name in sorted(on_disk - listed):
        fail(str(CONTRACTS_DOC.relative_to(REPO_ROOT)),
             "schema는 있는데 계약 목록에 없다: %s" % name)
    return len(listed | on_disk)


# --- 진입점 -------------------------------------------------------------------

def run(verbose: bool = False) -> int:
    failures: List[Tuple[str, str]] = []

    def fail(where: str, detail: str) -> None:
        failures.append((where, detail))

    fl = _check_fl_defaults(fail)
    gates = _check_gate_honesty(fail)
    contracts = _check_contract_set(fail)

    print("정책 검사: FL 기본값 %d·게이트 상태 %d·계약 집합 %d, 실패 %d건."
          % (fl, gates, contracts, len(failures)))
    if failures:
        for where, detail in failures:
            print("  %-44s %s" % (where, detail))
        print("이 변경은 자동 병합하지 않는다. 사람이 확인해야 한다.")
        return 1
    print("POLICY OK: fl=%d gates=%d contracts=%d failures=0" % (fl, gates, contracts))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
