# 데이터 어댑터 · B

어댑터는 미구현이다. `rosters/dunnhumby_rb.csv`는 기존 기준의 점포 ID 101개만 가진 재현용 목록이다. 고객·거래·배정표는 포함하지 않는다. 정제 원자료 재현은 아직 B의 작업이다.
첫 작업: 작은 Dunnhumby/Instacart/live 샘플을 공통 catalog/purchase 표현으로 변환하고 텍스트·결측·시간 의미를 확인한다. 실제 데이터·고객별 파생물은 Git에 넣지 않는다.
이전 탐색 분석 스크립트가 [fedcommerce/](../../../fedcommerce/README.md)에 있다. 현행 계약을 따르지 않는 참고 자료이므로 여기 어댑터는 계약에 맞춰 새로 쓴다.
공통 Python 타입은 `commerce.packages.contracts`, 모델 호출 진입점은 `commerce.packages.recommender.runtime`이다. 학습·평가 실행 코드는 `commerce/evaluation/`에 둔다.

기준과 한계: [데이터 설계](../../../docs/design/data.md), [B 작업 카드](../../../docs/team/tasks/B.md).
