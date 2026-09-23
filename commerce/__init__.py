# -*- coding: utf-8 -*-
"""commerce 패키지 루트.

저장소 루트를 작업 디렉터리로 두고 `python -m commerce.tools.gate <이름>`을 실행한다.
공개 인터페이스와 시작 방법: docs/development.md.

`commerce/packages/`에는 `__init__.py`를 두지 않는다. 그 디렉터리는 패키지 루트 모음이며
모든 서비스는 `commerce.packages.*` 절대 import를 사용한다. 별도 sys.path 수정은 하지 않는다.
"""
