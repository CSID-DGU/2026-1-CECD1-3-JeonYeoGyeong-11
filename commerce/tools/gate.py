# -*- coding: utf-8 -*-
"""단일 진입점 게이트 러너.

공개 계약·완료 상태: docs/contracts.md.

    python -m commerce.tools.gate <게이트명>

작업 디렉터리는 저장소 루트다. 종료 코드 0이 통과, 그 밖은 실패다.
게이트 목록과 실제 구현 여부는 이 파일과 docs/contracts.md에 함께 기록한다.

판정은 종료 코드 0과 기대 출력 마지막 줄의 동시 충족이다. `contracts` 게이트의
마지막 줄은 `CONTRACTS OK: schemas=<n> fixtures=<n> failures=0` 한 줄이며,
감싸 호출하는 `contracts.validate`가 같은 프로세스 stdout으로 낸다. 게이트가
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
    return _run


# 게이트명 -> (설명, 실행 함수). 공개 설명: docs/contracts.md.
GATES: Dict[str, Tuple[str, Callable[[bool], int]]] = {
    "contracts": ("모든 스키마와 fixture 검증(G1)", _run_contracts),
    "scaffold": ("공통 import·runtime 연결·기동 뼈대 검사", _run_scaffold),
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
