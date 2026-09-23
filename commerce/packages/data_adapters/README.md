# 데이터 어댑터 · B

어댑터는 미구현이다. 기존 로컬 `rosters/` 자료는 이번 초기 공통 코드에 포함하지 않는다.
첫 작업: 작은 Dunnhumby/Instacart/live 샘플을 공통 catalog/purchase 표현으로 변환하고 텍스트·결측·시간 의미를 확인한다. 실제 데이터·고객별 파생물은 Git에 넣지 않는다.
공통 Python 타입은 `commerce.packages.contracts`, 모델 호출 진입점은 `commerce.packages.recommender.runtime`이다. 학습·평가 실행 코드는 `commerce/evaluation/`에 둔다.
