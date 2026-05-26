# Previous Experiment Design Archive

## 목적

Raw update graph와 semantic client graph 방향의 초기 실험 설계를 보관한다.

## Design 요약

| Axis | 내용 |
|---|---|
| graph signal | raw update, local weight, EMA update |
| graph operator | cosine similarity, kNN, spectral smoothing |
| controls | random, shuffled, uniform, identity |
| metrics | accuracy, loss, graph metric, alignment |

## 전환

| 이전 초점 | 현재 초점 |
|---|---|
| raw update graph 성능 | graph gain attribution |
| spectral-only smoothing | matched control and diagnostic evidence |
| single graph variant | composable graph design space |

## 현재 연결

현재 실험 설계는 `docs/framework/overview.md`에 정리한다.
