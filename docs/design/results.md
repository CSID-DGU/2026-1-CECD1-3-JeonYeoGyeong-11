# B 결과 요약 — 데이터·NLP·추천
갱신: 2026-10-11 · 소유: B · 수치의 원본은 각 줄의 문서다. 원본이 바뀌면 이 쪽도 같이 고친다.

넘겨받는 사람이 한 장으로 볼 수 있게 B의 산출물, 주요 결과, 한계를 모은다. 정의와 근거는 원본 문서에 있다.

## 1. 만든 것

| 부분 | 내용 | 원본 |
| --- | --- | --- |
| 데이터 | Instacart·Dunnhumby 어댑터. 두 출처를 같은 `purchase_event.v1`·`catalog_item.v1`로 바꾼다. Instacart 고객을 판매자 100곳(가상)으로 나눈다 | [데이터](data.md) |
| NLP | 고정(frozen) 다국어 인코더 `paraphrase-multilingual-MiniLM-L12-v2`(384차원, 64토큰), 상품 텍스트 builder, 판매자 로컬 z cache | [NLP 인코더 선정](nlp-encoder.md) |
| 모델 | D0022 공통 뼈대에 사전학습 인코더 표현(lm). 서비스 variant는 text_only(T_lm)와 text_relation(R_lm, 구매 관계 추가) | [결정](decisions.md) D0024 |
| 판매자 runtime | 판매자 원장(features.sqlite), 추천, release 설치, FL 라운드 학습, 개인화, 네 결과 비교 | [모델 경계](model.md), `commerce/packages/recommender/README.md` |
| 실험 도구 | 비보호 FL 시뮬레이션(`fl_lab`), 단독 학습(`harex_compare`), 신규 판매자(`cold_start`, `a_few`), 결과표(`d0022_report`), 비용(`seller_cost`) | `commerce/evaluation/README.md` |

## 2. 주요 결과

Instacart 판매자 100곳, 비보호 FL 시뮬레이션(D0020), 다음 장바구니, NDCG@10 판매자 macro.

| 질문 | 결과 | 원본 |
| --- | --- | --- |
| 기존 판매자 (seed 3개) | T_hx 0.171 · R_hx 0.213 · T_lm 0.140 · R_lm 0.196 | D0024 |
| 신규 판매자, 학습 미참여(A-0, seed 3개) | T_hx 0.018 · R_hx 0.077 · T_lm 0.143 · R_lm 0.198. hx는 단어 표가 판매자 로컬이라 신규 판매자에게 쓸 수 없다 | D0024 |
| 구매 관계(R)의 효과 | 기존 판매자 R_lm − T_lm +0.056 [+0.051, +0.061]. 신규 판매자 R-G − T-G +0.041 [+0.025, +0.057] | D0024, [모델 비교](comparison.md) §9 |
| FL과 단독 학습 | R_lm 0.136(단독) → 0.196(FL), T_lm 0.102 → 0.140. 단독 학습이 자기 검증으로 checkpoint를 고르는 이점을 가지므로 보수적인 비교다 | [평가](evaluation.md) §5 |
| 판매자 개인화 | 검증을 통과한 판매자만 쓴다. 신규 판매자 10곳에서 채택 T 3곳·R 5곳, 채택된 곳에서 R +0.007 | comparison.md §9, model.md §8 |
| 판매자 한 곳의 비용(CPU, GPU 없음) | 라운드 학습 2.2초(T)·8.2초(R), 추천 40ms·9ms, 최대 메모리 1.8 GB | [독립 모델 실험](model-lab.md) §5 |
| 기준선 | 인기순 0.098, 고객 자기 구매 빈도(P-TopFreq) 0.403 | D0022 결과표 |

결론: 서비스는 R_lm(text_relation)을 기본으로 쓰고, 개인화는 검증을 통과할 때만 쓰는 작은 보정이다.

## 3. 서비스에 넘기는 것

- 첫 release 두 개: `ic100-T_lm-r500`, `ic100-R_lm-r500`(판매자 100곳, 500라운드). 가중치는 Git에 없고 팀 공유 드라이브의 `models.zip`에 있다(`SHA256SUMS.txt`로 확인). 출처 라벨은 "비보호 FL 시뮬레이션"이다.
- 인코더: `encoder_probe --download --only minilm-l12`로 고정 revision을 받는다. 설치할 때 해시가 다르면 거부한다.
- 설치: `python -m commerce.packages.recommender.install --model-dir … --encoder-dir … / --release-dir … --variant …`(FL 없이 서빙만 하는 판매자). FL에 참여하는 판매자는 C의 registry로 받는다.

## 4. 한계

- 실험실 결과는 보호 FL(G4)이 아니다. 비보호 FL 시뮬레이션으로 표시한다(D0020).
- 판매자는 Instacart 고객을 나눈 가상 판매자다. 실제 독립 기업이라는 근거로 쓰지 않는다([데이터](data.md)).
- 모델에 재구매 신호를 넣지 않았다(D0022·D0024). 그래서 고객 자기 구매 빈도 기준선(0.403)이 모델보다 높다.
- 작은 판매자·다음 상품 1개·신규 판매자 2×2는 seed 1개다. lm 계열 FL은 500라운드에서도 아직 수렴 전이었다.
- Dunnhumby는 상품명이 없어 점포 상품의 약 3분의 2가 텍스트가 같다(OQ04). 보조 cohort 결과는 따로 보고한다(진행 중).

## 5. 다시 만들기

- 실행 명령은 [평가](evaluation.md) §5에 있다. 결과표는 main 코드의 `python -m commerce.evaluation.d0022_report --runs commerce/evaluation/runs`로 만든다.
- 실행 기록(`commerce/evaluation/runs/`)과 원자료는 Git 밖에 둔다.
