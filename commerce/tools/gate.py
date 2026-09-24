# -*- coding: utf-8 -*-
"""단일 진입점 게이트 러너.

공개 계약·완료 상태: docs/contracts.md.

    python -m commerce.tools.gate <게이트명>

작업 디렉터리는 저장소 루트다. 종료 코드 0이 통과, 그 밖은 실패다.
게이트 목록과 실제 구현 여부는 이 파일과 docs/contracts.md에 함께 기록한다.

판정은 종료 코드 0과 기대 출력 마지막 줄의 동시 충족이다. 구현된 게이트의
마지막 줄은 각각 다음 하나다.

    CONTRACTS OK: schemas=<n> fixtures=<n> failures=0
    SCAFFOLD OK: tests=<n>; business gates are checked by gate business
    DOCS OK: links=<n> env=<n> gates=<n> imports=<n> vendor=<n> fences=<n> failures=0
    POLICY OK: fl=<n> contracts=<n> failures=0
    BUSINESS OK: passed=<n> not_implemented=<n> failures=0

실패하면 그 줄을 내지 않는다. 있는 그대로의 실패를 요약 문자열로 덮지 않기
위해서다. 각 줄은 감싸 호출하는 러너가 같은 프로세스 stdout으로 내며 게이트가
같은 줄을 다시 내지 않는다(중복 출력이면 마지막 줄 판정이 흐려진다).
한국어 요약 줄은 그 앞에 남는다.

업무 게이트(a1·b1·b2·c1·g2·g3·g4)는 아래 SELFCHECKS에 담당 경로의 모듈이 미리
연결돼 있다. 담당은 그 위치에 `run(verbose: bool = False) -> int`를 가진
selfcheck 모듈을 만들기만 하면 되고 이 파일을 고치지 않는다. 모듈이 없으면
NOT_IMPLEMENTED다. selfcheck의 종료 코드 규약:

    0  게이트의 모든 항목 통과
    3  구현한 항목은 통과, 남은 항목이 있음(남은 항목을 출력한다)
    1  실패한 항목이 있음
    2  검사 자체가 깨짐(모듈 import 실패 등)

CI가 모든 PR에서 돌리므로 게이트 항목은 lock과 저장소만으로 돌 수 있는 것만
둔다(작은 합성 입력, 작은 테스트 모델). 원자료·실제 가중치로만 확인되는 것(예:
b1의 원자료 재현)은 게이트 항목이 아니라 PR에 따로 보고한다. 그래서 항목을 모두
구현하면 CI에서도 0이 된다.

무엇을 확인해야 하는지는 docs/team/working-agreement.md §4가 기준이다.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import unittest
from typing import Callable, Dict, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
COMMERCE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(COMMERCE_DIR)
PACKAGES_DIR = os.path.join(COMMERCE_DIR, "packages")


def _run_contracts(verbose: bool) -> int:
    """계약 fixture 전수 검증. 개수는 러너가 동적으로 출력한다."""
    try:
        from commerce.packages.contracts import validate as contracts_validate
    except ImportError as exc:
        sys.stderr.write(
            "contracts 러너를 import하지 못했다: %s\n"
            "확인: %s 가 존재하고 jsonschema가 설치돼 있어야 한다.\n"
            % (exc, os.path.join(PACKAGES_DIR, "contracts", "validate.py")))
        return 2
    return contracts_validate.run(verbose=verbose)


def _run_docs(verbose: bool) -> int:
    """문서가 약속한 것과 코드에 있는 것의 대조. 개수는 러너가 동적으로 출력한다."""
    try:
        from commerce.tools import doccheck
    except ImportError as exc:
        sys.stderr.write(
            "docs 러너를 import하지 못했다: %s\n"
            "확인: %s 가 존재해야 한다.\n"
            % (exc, os.path.join(HERE, "doccheck.py")))
        return 2
    return doccheck.run(verbose=verbose)


def _run_policy(verbose: bool) -> int:
    """경로 규칙으로 잡히지 않는 위험 신호 검사."""
    try:
        from commerce.tools import policycheck
    except ImportError as exc:
        sys.stderr.write(
            "policy 러너를 import하지 못했다: %s\n"
            "확인: %s 가 존재해야 한다.\n"
            % (exc, os.path.join(HERE, "policycheck.py")))
        return 2
    return policycheck.run(verbose=verbose)


def _run_scaffold(verbose: bool) -> int:
    suite = unittest.defaultTestLoader.discover(
        os.path.join(COMMERCE_DIR, "tests", "e2e"), pattern="test_scaffold.py", top_level_dir=REPO_ROOT)
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    print("SCAFFOLD OK: tests=%s; business gates are checked by gate business" % result.testsRun)
    return 0


def _module_file(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _not_implemented(gate: str, module: str) -> Callable[[bool], int]:
    def _run(verbose: bool) -> int:
        print("NOT_IMPLEMENTED %s" % gate)
        print("추가 위치: %s (확인 항목: docs/team/working-agreement.md §4)" % _module_file(module))
        return 3
    return _run


def _selfcheck(gate: str, module: str) -> Callable[[bool], int]:
    try:
        present = importlib.util.find_spec(module) is not None
    except Exception:  # 한 역할의 패키지가 깨져도 다른 게이트는 import돼야 한다.
        present = True  # 실행할 때 그 오류를 그대로 보여 준다.
    if not present:
        return _not_implemented(gate, module)

    def _run(verbose: bool) -> int:
        try:
            check = importlib.import_module(module)
        except Exception as exc:
            sys.stderr.write("%s selfcheck를 import하지 못했다: %s\n" % (gate, exc))
            return 2
        return check.run(verbose=verbose)
    return _run


# 업무 게이트 -> 담당 소유 경로의 selfcheck 모듈. 위치를 여기서 미리 정해 두어
# 담당이 C 소유인 이 파일을 고치지 않고 자기 검사를 연결하게 한다.
SELFCHECKS: Dict[str, str] = {
    "a1": "commerce.services.merchant_api.selfcheck",
    "b1": "commerce.packages.data_adapters.selfcheck",
    "b2": "commerce.packages.recommender.selfcheck",
    "c1": "commerce.services.fl_coordinator.selfcheck",
    "g2": "commerce.tests.e2e.selfcheck_g2",
    "g3": "commerce.tests.e2e.selfcheck_g3",
    "g4": "commerce.tests.e2e.selfcheck_g4",
}


def _run_business(verbose: bool) -> int:
    """업무 게이트를 모두 실행한다. 미구현·부분 구현(3)은 허용하고 실패(1·2)는 막는다.

    CI가 이 게이트를 돌리므로 담당이 selfcheck를 추가하는 순간부터 그 검사가
    모든 PR에서 실행된다.
    """
    passed, pending, failed = [], [], []
    for gate in SELFCHECKS:
        code = GATES[gate][1](verbose)
        (passed if code == 0 else pending if code == 3 else failed).append("%s=%s" % (gate, code))
    print("업무 게이트: 통과 %d·미구현/부분 %d·실패 %d. %s"
          % (len(passed), len(pending), len(failed), " ".join(failed)))
    if failed:
        return 1
    print("BUSINESS OK: passed=%d not_implemented=%d failures=0" % (len(passed), len(pending)))
    return 0


# 게이트명 -> (설명, 실행 함수). 공개 설명: docs/contracts.md.
GATES: Dict[str, Tuple[str, Callable[[bool], int]]] = {
    "contracts": ("모든 스키마와 fixture 검증(G1)", _run_contracts),
    "scaffold": ("공통 import·runtime 연결·기동 뼈대 검사", _run_scaffold),
    "docs": ("문서 대 코드 대조(링크·환경변수·게이트명·import·도구 지시 파일·코드 펜스)", _run_docs),
    "policy": ("병합 전 위험 신호 검사(FL 기본값·계약 집합)", _run_policy),
    "business": ("업무 게이트 일괄 실행: 실패가 없으면 통과, 미구현·부분 구현은 허용", _run_business),
    "a1": ("A 주문·권한 업무 게이트", _selfcheck("a1", SELFCHECKS["a1"])),
    "b1": ("B 데이터·텍스트 입력 업무 게이트", _selfcheck("b1", SELFCHECKS["b1"])),
    "b2": ("B NLP·공유·개인화 업무 게이트", _selfcheck("b2", SELFCHECKS["b2"])),
    "c1": ("C 합성 라운드·모델 배포 업무 게이트", _selfcheck("c1", SELFCHECKS["c1"])),
    "g2": ("주문·실제 추천·비교 통합 게이트", _selfcheck("g2", SELFCHECKS["g2"])),
    "g3": ("합성 FL·신규 판매자 통합 게이트", _selfcheck("g3", SELFCHECKS["g3"])),
    "g4": ("보호 집계 통합 게이트", _selfcheck("g4", SELFCHECKS["g4"])),
}


def _usage_lines() -> str:
    out = ["사용 가능한 게이트:"]
    for name in GATES:
        out.append("  %-10s %s" % (name, GATES[name][0]))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="commerce.tools.gate",
        description="단일 진입점 게이트 러너. 설명: docs/contracts.md.",
        add_help=True)
    parser.add_argument("gate", nargs="?", help="게이트 이름")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="판정 1건마다 출력한다")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.gate:
        sys.stderr.write("게이트 이름이 없다.\n" + _usage_lines() + "\n")
        return 2

    if args.gate not in GATES:
        sys.stderr.write("알 수 없는 게이트: %s\n" % args.gate + _usage_lines() + "\n")
        return 2

    return GATES[args.gate][1](args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
