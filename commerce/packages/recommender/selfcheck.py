"""b2 selfcheck: NLP artifact and freeze, shared export, gradients, personalization limits.

Runs the tests in tests/ with a tiny random BERT, so CI needs no model download.
The real encoder is measured separately (docs/design/nlp-encoder.md). Exit 3
while items below remain.
"""
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

REMAINING = ()


def run(verbose: bool = False) -> int:
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), top_level_dir=str(REPO_ROOT))
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    print("b2 구현 항목 통과: tests=%d (frozen 인코더·해시·z cache, 예제·replay, 두 variant·관계·손실·공유 층 gradient, "
          "판매자 runtime: 특징 원장·추천 fallback·variant별 manifest·export·release 검증과 실패 복구·라운드 delta·"
          "고정 검증 고객 분할, 개인화: query_proj·scorer만 학습·base 불변·다른 base 부착 거부, 네 결과 비교)" % result.testsRun)
    for item in REMAINING:
        print("남은 b2 항목: %s" % item)
    return 3 if REMAINING else 0
