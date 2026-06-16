# 문서 안내

처음 보는 사람은 먼저 [README](../README.md)를 읽으면 된다. 이 폴더는 README에서 다루지 않은 세부 내용을 찾기 위한 문서 모음이다.

| 문서 | 내용 |
|---|---|
| [framework.md](framework.md) | component 구조, metric, artifact contract |
| [evidence.md](evidence.md) | 검증 결과와 한계 |
| [research.md](research.md) | prior work와 이 레포의 위치 |
| [repository.md](repository.md) | package, script, test 위치 |
| [maintenance.md](maintenance.md) | compatibility와 legacy 이름 처리 |
| [history.md](history.md) | 이전 실험 관찰과 정리 과정 |
| [demos/](demos/) | 보조 HTML demo |

## 빠른 확인

```powershell
python -m unittest discover -s tests
graphfl run single --track vision `
  --config configs/vision/smoke/default_similarity_knn.json `
  --dry-run
python scripts/checks/diagnostic_suite_preflight.py
```

문서의 명령과 수치는 실제 CLI와 검증 결과가 바뀌면 같이 갱신한다.
