# 보호 집계 선정 조사

갱신: 2026-10-01 · 소유: C · 실행 기준 [D0017·D0021](decisions.md)
상태: **조사 기록이며 채택 결정이 아니다.** 결론은 사람이 정한 뒤 [결정 요약](decisions.md)에 D 번호로 올린다. 아래 수치는 단일 프로세스 프로토타입의 측정이고 보호 검증(G4)의 증거가 아니다.

## 1. 요구와 범위

요구는 [D0021](decisions.md), [인터페이스](interfaces.md) §6~7, [아키텍처](architecture.md) §3에 있다. 이 문서는 되풀이하지 않고 선택에 영향을 준 것만 쓴다.

- 이탈 허용 0. 한 명이라도 빠지면 라운드를 폐기하므로 마스크 복구(비밀 분산)가 필요 없다.
- 새 암호 구성·자체 난수 생성기·자체 키 교환은 금지. 표준 라이브러리의 키 교환·키 유도·스트림 암호로 공개 프로토콜을 구현하는 것은 허용.
- 위협은 정상 프로토콜을 따르며 개별 정보에 관심을 갖는 중앙(honest-but-curious)이다. 악성 서버는 범위 밖이다.
- 환경은 Python 3.11, numpy 2.4.6, fastapi 0.141.1, uvicorn 0.53.0([lock](../../commerce/requirements-lock.txt)). 합성 평문 coordinator는 이미 FastAPI 위에 있다.

## 2. 후보 비교

| 후보 | 판정 | 근거 |
| --- | --- | --- |
| A. Flower `SecAggPlusWorkflow` (flwr 1.39.0) | 라이브러리로는 제외, 프로토콜 참조로만 | 아래 §2.1 |
| B. 공개 프로토콜(pairwise masking)을 pyca/cryptography로 조립 | **추천(사람 확인 필요)** | §2.2, §3 |
| C. TensorFlow Federated `SecureSumFactory` | 제외 | TF 생태계 연산자다. 이 프로젝트의 모델 경로는 PyTorch다. 단독 사용 가능 여부는 확인하지 않았다 |
| D. 동형암호 계열(예: NVIDIA FLARE의 HE 필터) | 이번에 검토하지 않음 | 마스킹 계열과 다른 방식이며 의존성과 암호문 크기가 크다. 비용을 측정하지 않았다 |
| E. 원 논문 전체(Bonawitz 2017: 이중 마스크·비밀 분산·서명) | 범위 밖 | 이탈 복구용 구성이다. D0021이 첫 범위에서 제외했다 |

### 2.1 Flower를 제외한 이유(확인한 것)

`pip install --dry-run flwr==1.39.0`을 프로젝트 venv에 대해 해석했다(설치하지 않음).

- `fastapi>=0.138.0,<0.139.0`, `uvicorn[standard]>=0.49.0,<0.50.0`, `starlette>=1.3.1,<1.4.0`을 요구한다. lock의 fastapi 0.141.1, uvicorn 0.53.0과 충돌하며 해석 결과는 둘을 내린다.
- 설치 대상이 30여 개 늘어난다(grpcio, protobuf, SQLAlchemy, alembic, uv, typer, rich 등).
- `cryptography>=46.0.7,<47.0.0`으로 상한을 둔다.
- SecAgg+ 워크플로는 Flower의 `ServerApp`/`ClientApp`과 SuperLink/SuperNode 전송 위에서 동작한다. 우리 coordinator는 자체 HTTP 경로와 Bearer 인증을 쓰므로 전송이 이중이 된다.
- Flower의 SecAgg는 이탈 복구를 위한 비밀 분산을 포함한다. 이탈 허용 0인 우리에게는 쓰지 않는 복잡도다.

따라서 의존성으로 들이지 않고, 메시지 구성과 상수를 읽는 참조 구현으로만 쓴다. 위협 모델(semi-honest)은 [Flower 논문](https://arxiv.org/abs/2205.06117)에 있다. 본 조사에서 그 문서의 세부(라운드 수 등)는 직접 확인하지 않았다.

### 2.2 pyca/cryptography로 조립하는 방식의 의존성

- `cryptography==50.0.2`는 추가로 `cffi`, `pycparser`만 끌어온다(dry-run으로 확인).
- Windows(`cp311-abi3-win_amd64`)와 Linux(`cp311-abi3-manylinux_2_28_x86_64`) wheel이 모두 있다. Linux CI에서 실제 설치·게이트 통과는 아직 확인하지 않았다.
- 필요한 기능은 `X25519`, `HKDF`, `ChaCha20`이며 import가 된다는 것을 확인했다.
- lock에 추가하는 것은 의존성 변경이므로 C가 환경 호환성을 확인하는 별도 PR로 한다([개발 안내](../development.md)).

## 3. 추천안: 이탈 복구 없는 pairwise masking

[Bonawitz 2017](https://eprint.iacr.org/2017/281.pdf)의 핵심인 쌍별 마스크만 쓰고, 이탈 복구에 쓰는 부분을 뺀다.

**라운드 흐름**

1. coordinator가 round_id, model_version, 고정 cohort(N명)를 안내한다.
2. 각 판매자가 그 라운드 전용 X25519 임시 키를 만들고 공개키를 coordinator에 보낸다. coordinator는 N명이 모이면 전체 공개키 목록을 돌려준다.
3. 판매자 i는 각 상대 j와 ECDH로 공유 비밀을 얻고 `HKDF-SHA256(salt=round_id, info=model_version|id_a|id_b)`로 키를 유도한다. 그 키로 ChaCha20 키스트림을 만들어 마스크 m_ij로 쓴다.
4. 판매자는 clip·양자화한 delta에 `+m_ij`(i<j) 또는 `-m_ij`(i>j)를 Z_2^32에서 더해 제출한다.
5. coordinator는 제출자가 cohort 전원인지 먼저 확인한다. 전원이면 합을 구해 평균을 공개하고 아니면 합을 계산하지 않고 라운드를 폐기한다.

**원 논문에서 뺀 것과 이유**

| 뺀 것 | 이유 | 이로 인한 한계 |
| --- | --- | --- |
| 비밀 분산(Shamir)·이탈 복구 | 이탈 허용 0(D0021) | 이탈하면 라운드 전체를 새 키로 다시 시작한다 |
| 개인 마스크(이중 마스킹) | 이중 마스킹은 "이탈했다고 주장된 사용자"의 쌍별 비밀을 서버가 복구할 수 있을 때를 막는다. 비밀 분산이 없으면 서버가 쌍별 비밀을 얻을 방법이 없다 | 아래 §6의 열린 항목 1 |
| 공개키 서명·합의 검사 | 악성 서버를 범위 밖으로 둠 | 키를 바꿔치는 중앙에는 보호되지 않는다(§6) |

세부 규칙은 다음과 같다.

- 키는 라운드마다 새로 만들고 라운드가 끝나면 폐기한다. 같은 쌍의 마스크를 다른 라운드에 재사용하지 않는다.
- 같은 라운드에서 같은 바이트의 재시도는 같은 결과를 돌려준다. 폐기된 라운드는 재개하지 않고 새 round_id와 새 키로 처음부터 한다([인터페이스](interfaces.md) §6과 같은 규칙).
- 같은 cohort에 대해 작은 부분집합의 합을 반복해서 공개하지 않는다. 부분집합 합은 만들지 않는다.
- 로컬 지표(`loss_mean` 등)와 유효 여부는 delta 벡터 뒤에 원소로 붙여 같은 방식으로 합산한다. 개별 값은 중앙에 보이지 않는다([인터페이스](interfaces.md) §7).

핵심 마스크 함수의 모양은 다음과 같다. 정확한 구현은 프로토콜 schema·fixture 뒤에 정한다.

```python
def mask(self, vec, peers, round_id, model_version):
    out = quantize(vec).astype(np.uint64)           # clip 후 고정소수점, Z_2^32
    for pid, peer_pub in peers.items():
        if pid == self.cid:
            continue
        key = self._pair_key(peer_pub, pid, round_id, model_version)  # X25519 -> HKDF
        m = chacha20_stream(key, vec.size).astype(np.uint64)           # 키스트림
        out = out + m if self.cid < pid else out + 2**32 - m
    return (out % 2**32).astype(np.uint32)
```

## 4. 프로토타입 측정

격리된 venv(Python 3.11.4, cryptography 50.0.2, numpy 2.4.6)에서 단일 프로세스로 실행했다. 프로젝트 코드가 아니며 저장소에 넣지 않았다. 네트워크·직렬화·인증 비용은 포함하지 않는다.

| 항목 | 결과 |
| --- | --- |
| N=5, 10만 원소, 평균 최대 오차 | 4.47e-7 (양자화 한계 0.5/2^20 = 4.77e-7 이내) |
| N=5에서 한 명 제외한 합 | 최대 오차 512. 값이 의미 없이 어긋나므로 합을 공개하면 안 된다 |
| 마스크된 벡터 vs 평문의 상관 | -0.002, 같은 값이 나온 비율 0 |
| 상위 4비트 분포 χ²/자유도 | 1.45. 균등과 어긋난다고 볼 근거는 없다. **균등함의 증명은 아니다** |
| 같은 입력, 다른 round_id | 마스크된 값이 같은 원소 0 |
| 판매자 1명의 마스크 계산 시간 | N=5·50만 원소 0.03 s, N=5·200만 원소 0.16 s, N=10·200만 원소 0.36 s |

- 한 명이 빠졌을 때 합이 틀리다는 것은 **합 자체로는 알 수 없다.** 따라서 제출자 수가 cohort와 같은지를 합산 전에 coordinator가 확인해야 한다(§3의 5단계).
- 비용은 판매자당 상대 수(N-1)에 비례한다.

## 5. 양자화와 크기

- 연산 공간은 Z_2^32다. 값은 `[-clip, clip]`으로 자른 뒤 `2^20`을 곱해 반올림한다.
- N명의 부호 있는 합이 int32를 넘으면 조용히 감긴다. 조건은 `N × clip × 2^20 < 2^31`이다. clip=4.0이면 N<512이고, 프로토타입은 이 조건을 넘는 N을 거부했다.
- 평균의 원소별 오차는 최대 `0.5/2^20`(약 4.8e-7)이다. clip을 넘는 값은 잘려 편향이 생기며, 이 오차는 별도다.
- **clip과 소수 비트는 아직 정하지 않았다.** B의 실제 delta 크기를 본 뒤 정한다([열린 구현 항목](open-questions.md)의 C 항목).
- uint32는 float32와 같은 4바이트다. 전송 한도 8 MiB([인터페이스](interfaces.md) §5)는 마스크 전에도 후에도 같다. 헤더를 제외하면 약 209만 원소가 상한이다.

## 6. 한계와 사람이 정할 것

**이 방식이 보장하지 않는 것(결과에 적는다)**

- 판매자 N-1명이 공모하면 나머지 한 명의 업데이트를 계산할 수 있다. 최소 인원 5명은 이를 없애지 않는다(D0021).
- 공개키를 중계하는 coordinator가 키를 바꿔치면 개별 업데이트를 복구할 수 있다. 키 서명이 없으므로 악성·침해된 중앙은 범위 밖이다. 서명하려면 판매자 키의 신뢰 경로가 필요하며 OQ14와 같은 문제를 만난다.
- 악의적인 판매자가 범위를 벗어나는 큰 값을 넣어도 마스크 때문에 중앙이 검사할 수 없다. clip은 판매자 쪽에서만 적용된다.
- 집계된 모델에서의 추론, 참여 여부 같은 운영 메타데이터는 보호하지 않는다([아키텍처](architecture.md) §2~3).
- 한 PC 시연은 호스트 관리자에 대한 격리를 증명하지 않는다.

**사람이 정할 항목**

1. **"이중 마스킹을 뺀 pairwise masking"이 D0021의 허용 범위인가.** 새 암호 구성이 아니라 공개 프로토콜에서 이탈 복구를 뺀 것이라는 것이 본 조사의 판단이지만, D0021이 애매하면 사람이 정하라고 했다. 개인 마스크 없이 안전한 근거는 위 §3 표와 같이 서버가 쌍별 비밀을 얻을 방법이 없다는 점이다. 이 논증은 외부 검토를 받은 것이 아니다.
2. `cryptography`를 lock에 추가할지. 추가하면 별도 PR에서 Windows·Linux CI 설치를 확인한다.
3. 키 바꿔치기를 범위 밖으로 둔 채 "보호 FL"이라 부를 수 있는 표현의 한계 문구.

## 7. 다음 작업(결정 뒤)

1. 위 1·2가 정해지면 프로토콜 메시지 schema와 fixture를 만든다([인터페이스](interfaces.md) §7). 계약 변경이므로 소비자 확인이 필요하다.
2. 합성 보호 집계를 `commerce/tests/e2e/`의 selfcheck로 만들고 실패·재시도(이탈, 중복 제출, round 불일치, 수 미달)를 검사한다.
3. 결정 전에도 되는 일: protected가 평문으로 조용히 떨어지지 않고 실패해야 한다([아키텍처](architecture.md) §3). 판매자 FL client는 이미 그렇다. `commerce/tests/e2e/test_scaffold.py`가 enabled FL을 protected·synthetic_plaintext 두 모드 모두에서 `FeatureNotImplemented`로 고정한다. coordinator 쪽 같은 고정은 c1 서비스 PR(브랜치 `commerce/c/c1-service`)이 `FL_MODE=protected` 기동 실패로 추가한다.
