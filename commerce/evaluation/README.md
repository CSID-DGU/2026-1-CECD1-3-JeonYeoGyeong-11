# 화면 없는 모델 실험 · B

현재 학습 실행기는 미구현이다. `commerce.packages.recommender`의 같은 모델 core를 사용하며 FastAPI 앱에 의존하지 않는다.
작업 순서는 [B 카드](../../docs/team/tasks/B.md)를 따른다. 손실 함수와 예제별 특징 cutoff는 본 학습 전에 확정한다([열린 구현 항목](../../docs/design/open-questions.md)).
산출물은 이 디렉터리의 Git 제외 `data/`, `cache/`, `runs/`, `outputs/`에 둔다. 집계 core는 C에게서 제공받으며 그 API는 선정 후 공동 확정한다. 모델 학습이 서비스 startup에서 실행되게 하지 않는다.
