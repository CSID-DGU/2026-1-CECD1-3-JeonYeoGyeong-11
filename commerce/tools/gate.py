# -*- coding: utf-8 -*-
"""단일 진입점 게이트 러너.

공개 계약·완료 상태: docs/contracts.md.

    python -m commerce.tools.gate <게이트명>

작업 디렉터리는 저장소 루트다. 종료 코드 0이 통과, 그 밖은 실패다.
게이트 목록과 실제 구현 여부는 이 파일과 docs/contracts.md에 함께 기록한다.

판정은 종료 코드 0과 기대 출력 마지막 줄의 동시 충족이다. 구현된 게이트의
마지막 줄은 각각 다음 하나다.

    CONTRACTS OK: schemas=<n> fixtures=<n> failures=0
    SCAFFOLD OK: tests=<n>; business gates remain NOT_IMPLEMENTED
    DOCS OK: links=<n> env=<n> gates=<n> imports=<n> vendor=<n> fences=<n> failures=0
    POLICY OK: fl=<n> gates=<n> contracts=<n> failures=0

실패하면 그 줄을 내지 않는다. 있는 그대로의 실패를 요약 문자열로 덮지 않기
위해서다. 각 줄은 감싸 호출하는 러너가 같은 프로세스 stdout으로 내며 게이트가
같은 줄을 다시 내지 않는다(중복 출력이면 마지막 줄 판정이 흐려진다).
한국어 요약 줄은 그 앞에 남는다.
"""

from __future__ import annotations

import argparse
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
    print("SCAFFOLD OK: tests=%s; business gates remain NOT_IMPLEMENTED" % result.testsRun)
    return 0


def _not_implemented(gate: str, need: str) -> Callable[[bool], int]:
    def _run(verbose: bool) -> int:
        print("NOT_IMPLEMENTED %s" % gate)
        print("필요: %s" % need)
        return 3
    # policy 게이트가 '실행 가능'이라는 문서 주장과 대조할 때 쓰는 표식이다.
    _run.not_implemented = True
    return _run


# 게이트명 -> (설명, 실행 함수). 공개 설명: docs/contracts.md.
GATES: Dict[str, Tuple[str, Callable[[bool], int]]] = {
    "contracts": ("모든 스키마와 fixture 검증(G1)", _run_contracts),
    "scaffold": ("공통 import·runtime 연결·기동 뼈대 검사", _run_scaffold),
    "docs": ("문서 대 코드 대조(링크·환경변수·게이트명·import·도구 지시 파일·코드 펜스)", _run_docs),
    "policy": ("자동 병합 전 위험 신호 검사(FL 기본값·게이트 상태 주장·계약 집합)", _run_policy),
    "a1": ("A 주문·권한·중앙 비잔류", _not_implemented("a1", "A 주문 API와 합성 상태 전이·권한·쿠키 검증")),
    "b1": ("B 데이터·텍스트 입력", _not_implemented("b1", "B data_adapters의 두 출처/live fixture와 텍스트 정규화 검증")),
    "b2": ("B NLP·공유·개인화 경계", _not_implemented("b2", "NLP artifact·freeze·variant별 export·gradient, 개인화 두 그룹 제한·base 불변·옛 tail 거부")),
    "c1": ("C 합성 라운드·모델 배포", _not_implemented("c1", "합성 제출 검증·균등 집계·모델 release·신규 판매자 설치·variant 혼합 거부")),
    "g2": ("주문·실제 추천·비교·장애 복구", _not_implemented("g2", "A/B 실제 연결·durable outbox 재전달, 비교 snapshot/후보 동일성·개인화 불가 표시")),
    "g3": ("합성 FL·개인화 전환·신규 판매자", _not_implemented("g3", "3판매자 실제 FL·base 교체 후 개인화 재생성·다른 상품 목록의 신규 판매자")),
    "g4": ("보호 집계·실패 경계", _not_implemented("g4", "선정 프로토콜의 개별 업데이트/손실 비노출·미달/이탈/재시도 검증")),
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
