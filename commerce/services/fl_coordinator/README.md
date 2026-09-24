# FL coordinator · C

진입점: `main:app`. 현재 health만 있고 제출·집계·모델 배포 API는 없다.
작업 순서는 [C 카드](../../../docs/team/tasks/C.md)를 따른다. 첫 라운드 경로는 생성 합성 텐서만 쓴다.
집계 core는 웹 앱과 분리해 B의 `commerce/evaluation/`에서도 재사용하도록 만든다. 보호 프로토콜 미확정 상태에서 실데이터 경로를 활성화하지 않는다.
