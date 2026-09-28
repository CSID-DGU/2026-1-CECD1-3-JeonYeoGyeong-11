"""b2 selfcheck: NLP artifact and freeze, shared export, gradients, personalization limits.

Runs the tests in tests/ with a tiny random BERT, so CI needs no model download.
The real encoder is measured separately (docs/design/nlp-encoder.md). Exit 3
while items below remain.
"""
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

REMAINING = (
    "variant별 shared_model_manifest와 실제 export(frozen·상품/고객축·optimizer 제외)",
    "공유 층 gradient와 고정 검증 분할(val seed에 round_id 없음)",
    "개인화: query_proj·scorer만 학습, 공통 base 해시 불변, 옛 개인화 부착 거부",
    "release 설치의 해시 검증과 교체 실패 시 기존 서빙 유지",
)


def run(verbose: bool = False) -> int:
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), top_level_dir=str(REPO_ROOT))
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    print("b2 구현 항목 통과: tests=%d (frozen 인코더·text_artifact_hash·z cache)" % result.testsRun)
    for item in REMAINING:
        print("남은 b2 항목: %s" % item)
    return 3 if REMAINING else 0
