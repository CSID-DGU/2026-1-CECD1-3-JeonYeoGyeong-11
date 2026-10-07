# NLP 인코더 선정 기록

갱신: 2026-09-28 · 소유: B · [B 카드](../team/tasks/B.md) 첫 작업 3번, [모델 경계](model.md) §2의 2번 산출물

frozen 텍스트 인코더의 선택과 근거를 적는다. [모델 경계](model.md)에는 결론만 별도 PR로 옮긴다. 아래 수치는 한 PC의 측정값이며, Dunnhumby 텍스트는 어댑터가 없어 아직 재지 않았다.

## 1. 결론

| 항목 | 값 |
| --- | --- |
| 모델 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| revision | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| 라이선스 | Apache-2.0 (모델 카드 태그) |
| tokenizer | 모델에 딸린 XLM-R SentencePiece(어휘 250,037). 소문자화하지 않는다. 새 special token을 더하지 않는다 |
| 입력 | [데이터](data.md) §3 builder의 텍스트 그대로. 접두어 없음 |
| 길이 | special token 포함 최대 64토큰, 넘으면 뒤를 자른다 |
| pooling | padding을 뺀 토큰 평균(모델의 `1_Pooling` 설정과 같음) 뒤 L2 정규화 |
| 차원 | d_text = 384, float32 |
| 설치 파일 | `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `sentencepiece.bpe.model`, `modules.json`, `sentence_bert_config.json`, `1_Pooling/config.json` |
| text_artifact_hash | `e84ab6c5207aa16046297e5e5d02c9d47319176c039a3ed04354370ba683660c` |

text_artifact_hash는 위 파일들의 SHA-256 목록과 인코더 설정의 정규 JSON 해시다([모델 경계](model.md) §2). 코드는 `commerce/packages/recommender/text_encoder.py`의 `artifact_hash`다. 이전 후보값 32토큰은 쓰지 않는다. Instacart 상품의 9.8%가 잘린다(§3).

## 2. 후보

| 후보 | revision | 라이선스 | 크기 | 판단 |
| --- | --- | --- | --- | --- |
| paraphrase-multilingual-MiniLM-L12-v2 | e8f8c21… | Apache-2.0 | 449 MiB, 384차원 | 채택 |
| intfloat/multilingual-e5-small | 614241f… | MIT | 449 MiB, 384차원 | 측정 후 제외. 무관한 상품끼리도 유사도가 높아 구분 폭이 좁고, 입력마다 `query: `가 붙어 잘림이 많다 |
| intfloat/multilingual-e5-base | d128750… | MIT | 1,061 MiB, 768차원 | 측정 전 제외. CPU 비용과 fusion 입력 차원이 두 배 이상이다 |
| BAAI/bge-m3 | 5617a9f… | MIT | safetensors 없음, 1,024차원 | 측정 전 제외. 판매자 PC의 frozen 인코더로 너무 크다 |
| jhgan/ko-sroberta-multitask | 8fca7c9… | 모델 카드에 라이선스 태그 없음 | 422 MiB | 측정 전 제외. 라이선스를 확인할 수 없고 영어 원자료에 맞지 않는다 |

## 3. 측정

조건: Windows 11, Intel Core(Family 6 Model 191, 16 논리 코어), torch 2.13.0+cpu 10스레드, Python 3.11.9. 입력은 Instacart 상품 49,688개 전체(원자료 로컬 파일, 집계만 기록)와 직접 지은 한국어 probe 13종이다. 두 후보 모두 파라미터 117,653,760개가 전부 학습 불가 상태였고, 같은 입력을 두 번 넣으면 같은 벡터가 나왔다.

| 항목 | MiniLM-L12 | e5-small |
| --- | --- | --- |
| unknown 토큰 비율 (Instacart / 한국어) | 0 / 0 | 0 / 0 |
| 디코드 결과가 원문과 같은 비율 (Instacart) | 99.86% | 99.84% |
| 토큰 길이 중앙 / 95% / 최대 (Instacart) | 27 / 34 / 62 | 30 / 37 / 65 |
| 잘리는 상품 @32 / @48 / @64 | 9.8% / 0.09% (47개) / 0 | 24.8% / 0.21% / 1개 |
| 잘린 뒤 다른 텍스트와 입력이 같아지는 상품 @32 | 0 | 9개 |
| 한국어 probe 최대 토큰 | 38 | 41 |
| 코사인: 규격만 다름 (한 / 영) | 0.979 / 0.991 | 0.984 / 0.993 |
| 코사인: 유기농·일반 우유 | 0.957 | 0.989 |
| 코사인: 무가당·가당 (한 / 영) | 0.995 / 0.966 | 0.983 / 0.988 |
| 코사인: 한영 같은 상품 | 0.961 | 0.904 |
| 코사인: 같은 코너 다른 상품 (우유·요거트) | 0.738 | 0.919 |
| 코사인: 무관 (한 / 영) | 0.627 / 0.426 | 0.884 / 0.887 |
| 처리량 (Instacart 5,000개, 번갈아 3회) | 약 229개/초 | 약 209개/초 |
| 프로세스 최대 메모리 (Instacart 전체 인코딩까지) | 1,293 MiB | 1,495 MiB |

처리량의 첫 측정은 MiniLM 107개/초로 낮게 나왔다. 같은 조건에서 두 모델을 번갈아 다시 재니 위 값으로 수렴해서, 첫 값은 측정 편차로 본다. 이 CPU는 성능·효율 코어가 섞여 있어 실행마다 차이가 날 수 있다. Instacart 상품 전체의 z는 약 4분이면 만들고, z cache가 있으면 텍스트가 바뀐 상품만 다시 계산한다.

## 4. 해석과 한계

- 규격·당 함량만 다른 상품은 z가 매우 가깝다. 특히 "무가당 두유"와 "가당 두유"는 0.995다. 구별 가능하지만 차이가 작아서, 관계 특징과 학습되는 fusion이 보완해야 한다. 이 텍스트만으로 구별되지 않는 정도는 OQ04에서 따로 보고한다.
- **텍스트가 같은 상품(OQ04).** 모델에는 상품 ID·bias가 없어서 인코더 입력이 같으면 z가 같고, 구매 관계도 없으면 점수가 같다. `commerce/evaluation/text_collisions.py`로 쟀다(2026-10-07). 한 판매자의 순위는 그 판매자 카탈로그 안에서 정하므로 판매자 안의 비율이 실제 영향이다.

  | 출처 | 상품 전체 | 판매자 카탈로그 안 (평균 / 최대) | 64토큰에서 잘림 |
  | --- | --- | --- | --- |
  | Instacart (100곳 cohort가 산 상품 37,459개) | 0.12% | 0.04% / 0.16% | 0% |
  | Dunnhumby (roster 93곳이 산 상품 50,325개) | 85.0% | 62.9% / 72.4% | 0% |
  | live fixture | — | 0% | 0% |

  - 잘리는 상품이 없어서 token 단계에서 새로 같아지는 상품은 없다. 정규화한 텍스트가 같은 경우만 같다.
  - Instacart는 영향이 거의 없다.
  - Dunnhumby는 한 점포 상품의 약 3분의 2가 같은 점포의 다른 상품과 텍스트가 같다. text_only는 이들을 구별하지 못하고, text_relation은 구매 관계로만 구별한다. 관계도 없는 상품(신상품, cutoff 뒤에만 팔린 상품)은 같은 점수다. 그래서 DH는 보조 cohort이고 결과를 정답 상품의 텍스트 고유 여부로 나눠 보고한다([평가](evaluation.md) §4).
- 같은 텍스트라도 함께 묶인 배치의 패딩 길이에 따라 z가 1e-8 수준으로 다를 수 있다. 계산 순서가 달라지기 때문이다. z cache가 처음 계산한 값을 재사용하므로 학습과 서빙은 같은 z를 본다. 따로 계산한 z끼리 비교할 때의 허용오차는 절대값 1e-6으로 둔다.
- 코사인의 절대값은 모델마다 기준이 다르다. 여기서는 순서와 폭을 봤다. MiniLM은 규격 차이 > 한영 같은 상품 > 같은 코너 다른 상품 > 무관 순서가 뚜렷하다.
- live 상품은 [DESC]가 길면 64토큰에서 뒤쪽 [CAT]이 잘린다. builder의 필드 순서([데이터](data.md) §3)를 [NAME]·[CAT]·[DESC]로 바꾸면 설명만 잘린다. live 상품 설명의 길이를 본 뒤 따로 정한다.
- Dunnhumby 텍스트의 충돌은 위 표와 같다. 같은 텍스트는 어느 인코더로도 같은 벡터가 되므로 이 때문에 인코더 선택을 다시 열지는 않는다. 약어형 텍스트([CAT]·[TYPE]·[SIZE])의 의미 품질(코사인 순서)은 따로 재지 않았다. DH는 보조 cohort다.

## 5. 모델 경계에 옮길 것

[모델 경계](model.md) §2의 "384차원·32토큰·MiniLM급은 이전 후보값" 문장을 위 결론(모델 ID·revision, 64토큰, d_text=384)으로 바꾸는 PR을 따로 올린다. CODEOWNERS 경로라 다른 역할의 승인이 필요하다. manifest의 fusion·relation_mlp 입력 차원은 d_text=384를 쓴다.

## 6. 재현

모델 파일을 `commerce/evaluation/cache/encoders/<org>__<name>/<revision>/`에 받은 뒤 실행한다. 결과 JSON은 Git 제외인 `commerce/evaluation/outputs/encoder_probe/`에 쌓인다.

```text
python -m commerce.evaluation.encoder_probe --instacart-dir fedcommerce/data/instacart --only minilm-l12
python -m commerce.evaluation.encoder_probe --instacart-dir fedcommerce/data/instacart --only e5-small
```
