# 평가 지표와 비모델 기준선 · A

정의는 [평가](../../../docs/design/evaluation.md) §4다. 모델을 만들지 않는 A가 채점을 맡는다([작업 규칙](../../../docs/team/working-agreement.md) §6). 원자료나 특징 DB는 읽지 않고, B 실행기가 예제마다 넘기는 점수·정답·구매 횟수만 받는다.

`ranking.py`:

| 이름 | 하는 일 |
| --- | --- |
| `expected_metrics(scores, relevant, ks=(10, 20))` | 후보 전체 점수 배열과 정답 후보 인덱스로 Recall@K·NDCG@K. 동점 구간은 순서를 정하지 않고 기대값으로 계산한다 |
| `popularity_scores(items, seller_counts)` | 로컬 인기순: cutoff 이전 판매자의 상품별 구매 횟수 |
| `p_topfreq_scores(items, prior_counts, seller_counts)` | P-TopFreq: cutoff 이전 그 고객의 상품별 구매 횟수, 동점은 로컬 인기순 |
| `repeat_explore_indices(items, relevant, prior_counts)` | 정답을 재구매(repeat)·첫 구매(explore)로 나눈다 |
| `new_item_indices(items, relevant, seller_counts)` | 정답 중 cutoff 이전 판매 이력이 없는 상품(신상품 cohort) |
| `MacroAverager` | `add(seller, customer, metrics)` → `result()`: 고객별 → 판매자별 → 판매자 macro 평균, 전체 예제 micro 평균, 유효 query 수(`examples`). 보고 구간(전체·repeat·explore·신상품)마다 하나씩 쓴다 |
| `recall_at_k`·`ndcg_at_k`·`local_popularity_ranking`·`p_topfreq_ranking` | 같은 정의의 (item_id, score) 목록 버전. 판매자 앱의 임시 추천이 쓴다 |

B의 임시 `commerce/evaluation/scoring.py`와 함수 이름·인자가 같다. `from commerce.evaluation.scoring import ...`를 `from commerce.evaluation.metrics.ranking import ...`로 바꾸면 된다. `tests/test_ranking.py`가 손계산 예제와 함께, `scoring.py`가 있는 동안 무작위 예제 500개(동점 많음)·기준선·평균에서 두 구현이 소수 12자리까지 같은지 확인한다. `scoring.py`가 지워지면 그 교차 검사는 건너뛴다.

재구매·첫 구매·신상품 구간은 정답이 비면 그 예제를 그 구간에서 뺀다(`expected_metrics`는 정답 없는 예제를 거부한다).
