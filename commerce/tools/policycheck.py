# -*- coding: utf-8 -*-
"""경로 규칙으로 잡히지 않는 위험 신호 검사.

    python -m commerce.tools.gate policy

CODEOWNERS는 "어떤 파일을 고쳤는가"만 본다. 그런데 소유자가 자기 경로 안에서
FL를 켜거나 계약 schema를 빼는 변경은 경로만 보면 평범하다. 이 검사는 그런 상태를
직접 확인해 병합을 막는다.

판정은 종료 코드 0과 마지막 줄
`POLICY OK: fl=<n> contracts=<n> failures=0`의 동시 충족이다.
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
    return checked + _check_fl_enablers(fail)


# --- 1b. FL을 켜는 코드는 합성 시연 실행기 한 곳뿐이다 (D0025, OQ16) ---------------

FL_DEMO = REPO_ROOT / "commerce/deploy/fl_demo.py"


def _is_test(path: pathlib.Path) -> bool:
    return "tests" in path.parts or path.name.startswith("test_")


def _check_fl_enablers(fail) -> int:
    """FLClientConfig에 상수 enabled=True나 synthetic_attestation을 넘기는 곳은 fl_demo.py뿐이다.

    synthetic_plaintext는 확인 파일(synthetic_attestation)이 있어야만 학습·제출한다. 그래서 그
    인자를 넘기는 곳이 FL을 실제로 켜는 곳이다. 설치 전용(install_only)은 제출하지 않지만
    coordinator에 접속하므로 같은 규칙으로 묶는다. 거기서는 mode가 상수 "synthetic_plaintext"여야 한다.
    판매자 앱의 환경변수 경로(FL_ENABLED)는 확인 파일을 넘기지 않으므로 켜도 시작이 거부된다.
    """
    checked = 0
    for path in sorted((REPO_ROOT / "commerce").rglob("*.py")):
        if _is_test(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None))
                    == "FLClientConfig"):
                continue
            keywords = {k.arg: k.value for k in node.keywords if k.arg}
            enabled = keywords.get("enabled")
            install_only = keywords.get("install_only")
            turns_on = ("synthetic_attestation" in keywords
                        or (isinstance(enabled, ast.Constant) and enabled.value is True)
                        or (isinstance(install_only, ast.Constant) and install_only.value is True))
            if not turns_on:
                continue
            checked += 1
            where = "%s:%d" % (path.relative_to(REPO_ROOT).as_posix(), node.lineno)
            mode = keywords.get("mode")
            if path != FL_DEMO:
                fail(where, "FL을 켜는 FLClientConfig는 commerce/deploy/fl_demo.py에만 둔다(D0025)")
            elif not (isinstance(mode, ast.Constant) and mode.value == "synthetic_plaintext"
                      and ("synthetic_attestation" in keywords or "install_only" in keywords)):
                fail(where, "fl_demo.py의 FLClientConfig는 mode=\"synthetic_plaintext\"와 "
                            "synthetic_attestation(또는 설치 전용 install_only)을 함께 쓴다")
    return checked


# --- 2. 계약이 조용히 사라지지 않아야 한다 ------------------------------------

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
    contracts = _check_contract_set(fail)

    print("정책 검사: FL 기본값 %d·계약 집합 %d, 실패 %d건." % (fl, contracts, len(failures)))
    if failures:
        for where, detail in failures:
            print("  %-44s %s" % (where, detail))
        print("이 변경은 병합하지 않는다. 담당자가 위 항목을 고친다. 정책 자체를 바꿔야 할 때만 사람이 결정한다.")
        return 1
    print("POLICY OK: fl=%d contracts=%d failures=0" % (fl, contracts))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
