

초기에는 선행연구 조사가 충분하지 않은 상태에서 client graph + spectral smoothing 방향의 novelty를 크게 봤습니다.

최근 관련 연구를 다시 확인해보니, 이 방향은 이미 많이 연구되어 있다는 점을 확인했습니다. 이 부분은 제가 처음에 조사가 부족했던 것 같습니다. 죄송합니다. 그래서 주장의 방향을 수정할 필요가 있을 것 같습니다.

다만 프로젝트 자체를 접어야 한다는 뜻은 아닙니다. 지금까지 구현한 FedAvg/FedAvgM baseline, diagnostics logging, graph control 실험, random/shuffled/uniform/identity 비교, conflict/cancellation/dominance 분석은 그대로 활용할 수 있을 것 같습니다.

그래서 기존 방향을 완전히 버리기보다는, 주장을 조금 더 좁히는 쪽이 맞는 것 같아요. 단순히 “client graph를 만들고 smoothing해서 성능을 올린다”가 아니라, Non-IID FL에서 client update들이 서로 충돌하거나, 평균 과정에서 상쇄되거나, 일부 client가 과도하게 영향을 주는 현상을 분석하는 방향입니다.

즉 graph smoothing 자체를 폐기하자는 것이 아니라, 먼저 어떤 client relation이 실제로 aggregation failure를 설명하는지 확인하고, 그다음에 smoothing/filtering이 유효한지 검증해보자는 의미입니다.

수업 프로젝트 관점에서는 지금까지의 구현과 실험을 바탕으로 문제 정의, 선행연구 조사, 실험 결과와 해석, 향후 개선 방향까지는 충분히 정리해볼 수 있을 것 같습니다.

---

**이 글에서 negative result를 이렇게 읽어 달라**  
실패한 것으로 보이는 결과는 곧바로 **“그래프 기반 교정이라는 카테고리가 원리적으로 막혔다”**는 뜻이 **아니다**. 동일한 수치는 **(A) 엣지를 만든 representation이 병리(충돌·상쇄·지배)를 거의 담지 못한 그래프**였거나, **(B) 그래프와 무관하게 Laplacian 등 operator가 global update \(\Delta\)를 사실상 바꾸지 못했거나**, **(C) 클라이언트 수·라운드·α 등 설정이 너무 가혹해 신호 대비 노이즈가 컸을 때**도 설명된다. 현재 증거는 **(A)와 (B)를 아직 깔끔히 분리하지 못한 상태**이며, 본 문서 후반 §6·§7은 그 분리와 재시험을 위한 순서를 적어 둔다.

---

## 0. 전체 요약

> 우리는 새로운 graph smoothing 알고리즘을 강하게 주장하기보다, label-skew single-global FL에서 client interaction을 conflict, cancellation, dominance **diagnostic axes**로 읽고 failure를 설명·검증할 수 있는지 보며, 어떤 client representation이 **pathology-aware graph**로 통제 실험에서 유효한지 검증하는 방향으로 연구 범위를 좁힌다.

**1) 사전연구 때문에 버리거나 어렵다고 보는 것**

- graph smoothing/filtering 자체를 **폐기할 이유는 없다**(방법 카테고리는 이미 여러 각도로 존재한다).
- 다만 **“client graph + Laplacian/graph smoothing으로 FL aggregation을 개선한다”**는 문장 하나를 **넓은 novelty**로 내세우기는 어렵다. 이유는 **client relation / personalized aggregation**(pFedGraph, FedAMP, pFedSim, FedAGA 등), **서버 단 conflict 교정**(FedGH, ConFREE 등), **norm reduction·cancellation**(FedMRUR 등), **dominance·aggregation bias**(GMA, FedHEAL 등), **GSP/graph filtering**(G-Fedfilt 등)이 이미 가까운 축을 점유하고 있기 때문이다.
- 정리하면, **안 되는 것은 smoothing의 존재**가 아니라 **그것만으로 “우리가 처음”이라고 말하는 claim**이다.

**2) 현재 실험 결과 때문에 약하다고 볼 수 있는 것**

- **smoothing 전체가 아니라**, 아래 **조합에 한정**된 negative signal이다.  
  instant/one-round representation · cosine similarity · **positive-edge graph** · **unnormalized Laplacian** · 작은 client 수 · 극단 label skew.
- 먼저 짚을 **주된 해석 후보**는 다음이다. **① 그래프가 “있어 보이는” 구조만 주고 pathology를 거의 안 실었을 가능성**(random/shuffled control과 차이가 잘 안 나는 패턴과도 맞먹음). **② 같은 이유로 operator에 넣어도 \(\Delta\) 방향이 거의 안 바뀐 것**(Phase 2.5-A의 cos≈1). **③ 설정·표본 크기 때문에 효과가 가려졌을 가능성**(별도 검증 필요).
- Phase 2/2.5에서 이 조합은 FedAvgM 및 graph control 대비 우위가 불명확했고, 2.5-A에서는 corrected update 방향 변화도 제한적(cos≈1)이었다.
- 따라서 문장으로는 이렇게 고정한다. “smoothing 자체가 이론적으로 불가능하다”는 말은 하지 않는다. 대신 **“시험한 그래프는 병리 신호를 충분히 담지 못했을 수 있고, 그 위에 얹은 operator 조합도 \(\Delta\)를 충분히 움직이지 못했을 수 있다”**고 말한다.

**3) smoothing/filtering 방향이 아직 열려 있는 이유**

- 동일 증거로는 **더 나은 representation**(EMA·head·signed conflict 등), **signed/normalized·random-walk·residual** 등 operator, 그리고 **graph control + graph-free baseline**과의 대조까지 거치지 않은 상태다.
- 다음 단계는 **폐기가 아니라 재검증**: proper graph construction, signed/normalized operator, residual mixing, graph controls, graph-free baseline 비교를 같은 **diagnostic protocol** 안에 두는 것.

**결론**

사전연구 때문에 graph smoothing 자체를 버리는 것은 아니다. 다만 graph smoothing을 넓은 novelty claim으로 내세우기는 어렵다. 현재 실험 때문에 smoothing 전체를 버리는 것도 아니다. 다만 지금의 instant positive graph + unnormalized Laplacian 조합은 약했다. 따라서 다음 단계는 smoothing 폐기가 아니라, proper graph construction, signed/normalized operator, graph controls, graph-free baseline 비교를 갖춘 재검증이다.

**실험 우선순위**

1. **§7.0 최소 1차 ablation**을 먼저 돌린다(RQ2·RQ3의 핵심 비교를 한 번에 통과시키는 최소 세트).
2. **Operator sanity(§7.4)**는 graph source가 “완전히” 검증된 뒤가 아니라, **현재 operator가 Δ를 실제로 바꿀 수 있는지** 확인하기 위해 **별도로 앞에서도** 수행한다. 다만 **성능 개선 목적의 넓은 operator tuning**은 informative graph가 잡힌 뒤 확장한다.
3. graph가 살아난 뒤 **RQ4**로 normalized/random-walk/residual/signed 등을 체계적으로 넓힌다.

**바로 피해야 할 과장**(상세 목록은 **§9.3**)

- “새 graph smoothing 방법이 SOTA급이다”, “harmful interaction을 우리가 처음”, “topology가 증명됐다”, “dominance 교정으로 정확도 개선 증명” 등은 현재 증거·선행과 맞지 않는다.

---

## 1. 왜 방향을 바꾸는가

처음 설정은 대략 아래 블록에 가깝다.

```text
client update/model로 그래프를 만들고
smoothing/filtering으로 global aggregation을 개선한다
```

이 문장 하나만 들고 나가면 **personalized collaboration graph**(pFedGraph, FedAMP, FedAGHN 등), **dynamic client relation**(FedAGA 등), **G-Fedfilt류 GSP 필터링**, **서버 단 conflict 보정**(FedGH, ConFREE), **norm reduction**(FedMRUR), **dominance/bias**(GMA, FedHEAL 등)와 겹친다.

그래서 **novelty는 “graph를 쓴다”가 아니라** 다음처럼 쪼개져야 한다.

```text
- 무엇이 failure인가?  → conflict / cancellation / dominance diagnostic axes
- relation이 무엇을 설명하는가? → collaboration이 아니라 aggregation distortion
- graph가 정말 신호인가? → random/shuffled/uniform/identity 통제 후에만 논함
```

**설계 순서도 바꿨다고 보는 것이 맞다.** (i) 어떤 client representation/graph construction이 pathology를 담는지 **먼저**,(ii) control 대비 우위 확인,(iii) **그 다음** smoothing/filtering/correction. Phase 2/2.5는 (iii) 쪽을 밀면서 (i)를 덜 고정한 면이 있다.

선행연구가 “구성요소별로는 이미 강하다”는 점은 **§5와 §9.1 표**에서 한 번만 정리한다 — 여기서는 반복하지 않는다.

---

## 2. 현재 실험에서 확인된 것

설정 공통점: FashionMNIST, Dirichlet 라벨 skew **α=0.03**, **clients=5**, seeds **42,43,44**, rounds **10**(Phase 2.5-A는 통계만 별도 기술).

### 2.1 Phase 1: pathology 로그와 baseline

| Method | Mean final accuracy |
|---|---:|
| FedAvg | 0.7700 |
| FedAvgM | 0.8111 |

| Metric | FedAvg | FedAvgM |
|---|---:|---:|
| CR (conflict ratio) | 0.6733 | 0.4400 |
| CA (cancellation) | 0.5607 | 0.5040 |
| DI (dominance) | 0.3058 | 0.3120 |
| N_eff | 4.3286 | 4.2834 |

참고로 seed 3개에서 `corr(final_acc, mean_DI)=-0.5984` 같은 수치가 나오기도 했으나, **표본 수가 극히 작아 correlation 값 자체를 claim으로 쓰면 안 된다.** Phase 1은 **pathology metric이 계산 가능하고**, 일부 run에서 성능 로그와 **같은 방향으로 움직일 수 있음을 보여 준 pilot / logging sanity check**에 가깝다. **강한 evidence가 아니라 파이프라인 동작 확인** 정도로 본다.

### 2.2 Phase 2: naive graph smoothing (λ=0.05)

| Variant | Mean accuracy |
|---|---:|
| update graph | 0.8008 |
| random | 0.8095 |
| shuffled | 0.8103 |
| uniform | 0.8087 |
| identity | 0.8108 |
| FedAvgM baseline | 0.8123 |

### 2.3 Phase 2.5-A (A_runs = 27)

| run id | mean acc | std |
|---|---:|---:|
| A_fedavgm_baseline | **0.8135** | **0.0148** |
| A_lap_l0p2_identity | 0.8108 | 0.0143 |
| A_lap_l0p2_shuffled | 0.8037 | 0.0198 |
| A_lap_l0p2_uniform | 0.8003 | 0.0298 |
| A_lap_l0p2_update | 0.8124 | 0.0141 |
| A_lap_l0p5_identity | 0.8108 | 0.0143 |
| A_lap_l0p5_shuffled | 0.8056 | 0.0212 |
| A_lap_l0p5_uniform | 0.8033 | 0.0202 |
| A_lap_l0p5_update | 0.8058 | 0.0224 |

방향 불변 프록시(cos≈1, ∥Δ_rel∥는 λ=0.5에서 상대적으로 커짐): **표 전체는 §9.2**(원본 수치)·보고서에 두고, 해석만 §6에서 한다.

### 2.4 Phase 3 (메모 한 줄)

norm clipping 등으로 DI/N_eff가 **조금** 움직인 정도까지는 보고됐지만, FedAvgM·uniform·cap·soft dominance가 한 표에 묶이기 전에는 **교정 방법 비교 결론을 내리지 않는다**.

---

## 3. 수정된 연구 질문

중심 전환은 유지한다.

```text
from: who should collaborate?
to:   who distorts aggregation?
```

이를 아래 네 가지 연구 질문으로 고정한다.

- **RQ1.** label-skew single-global FL에서 conflict, cancellation, dominance **diagnostic axes**는 aggregation failure를 설명하는 데 도움이 되는가?
- **RQ2.** 어떤 client representation이 random/shuffled/uniform/identity control보다 informative한 client interaction graph를 만드는가?
- **RQ3.** 그 graph는 CR/CA/DI/N_eff 같은 scalar diagnostic만으로 설명되는 수준인가, 아니면 **실제 추가 정보**(교정·잔차·라운드별 패턴 등)를 주는가? 동시에 **graph-free correction**(uniform weighting, norm clipping, contribution cap, soft dominance reweighting) 대비 무엇이 나은가?  
  **RQ3 없이 진행하면** 연구가 “metric logging + 실패한 smoothing”으로 보일 위험이 있으므로, **차별점은 graph smoothing 공식이 아니라 controlled protocol 위에 있다고 보는 것**이 맞다.
- **RQ4.** graph가 유효하다면, normalized smoothing, residual mixing, signed conflict correction, dominance-aware attenuation 중 어떤 operator가 실제 global update 방향과 pathology metric을 바꾸는가?

즉 순서는 **representation → control 비교 → RQ3(추가 가치)·graph-free 대비 → RQ4 operator**다.

---

## 4. Aggregation Pathology 정의

round \(t\), client \(i\) 업데이트 \(g_i^t = w_i^{local,t} - w^t\). 가중 global 업데이트 \(\Delta_t = \sum_i p_i g_i^t\).

conflict, cancellation, dominance는 aggregation failure를 **완전히 분해하는 taxonomy**라기보다, label-skew single-global FL에서 aggregation 상태를 읽기 위한 **diagnostic axes**로 둔다. 세 축은 **서로 독립된 failure mode라고 단정하면 안 된다**(예: conflict가 cancellation을 키우거나, dominance가 cancellation 형태와 함께 bias를 만들 수 있다). 따라서 분석에서는 **축 간 상호작용**을 허용하는 **failure analysis framework** 안에서 로깅하고 해석한다.

### 4.1 Conflict

client update 간 파괴적 방향 불일치:

\[
\cos(g_i^t,g_j^t)<0,\qquad
CR_t =
\frac{|\{(i,j):\cos(g_i^t,g_j^t)<0\}|}{\#\text{pairs}}
\]

edge 대안: \(C_{ij}=\mathbf{1}[\cos<0]\) 또는 \(C_{ij}=[-\cos]_+\).

### 4.2 Cancellation

개별 노름은 있는데 평균 후 global 노름이 작아지는 현상:

\[
CA_t =
1-
\frac{\|\sum_i p_i g_i^t\|}{\sum_i p_i\|g_i^t\|+\epsilon}
\]

### 4.3 Dominance

일부 클라이언트가 합성 방향을 과도하게 끈다.

\[
q_i=p_i\|g_i^t\|,\quad
\bar q_i=\frac{q_i}{\sum_j q_j+\epsilon},\quad
DI_t=\max_i \bar q_i
\]

\[
N_{\text{eff},t}=1/\bigl(\sum_i \bar q_i^2+\epsilon\bigr)
\]

---

## 5. 선행연구와의 관계

**축별로 이미 많이 다뤄진 것**

| 축 | 가까운 예 | 참고 포인트 |
|---|---|---|
| client relation · collaboration graph · personalized aggregation | **pFedGraph**, **FedAMP**, **pFedSim**, **FedAGA**(및 유사 라인) | 목적은 대개 “누구와 협력해 개인화할 것인가”. 우리는 **single-global**에서의 failure·topology **추가 가치**를 주장하려면 **통제 프로토콜**으로 분리해야 함. |
| server-side gradient conflict 교정 | **FedGH**(SPL Gradient Harmonization), **ConFREE** 등 | 우리 문장 하나 “충돌을 줄인다”만으로 차별 불충분. |
| norm reduction · cancellation 형태 | **FedMRUR** 등 | cancellation 언어가 겹침 → “처음 논했다” 불가. |
| dominance · aggregation bias | **GMA**, **FedHEAL** 등 | domain vs label-skew 문제 설정 차이 명시 필요. |
| GSP · graph filtering FL | **G-Fedfilt** 등 | smoothing 언어는 겹치나 graph source 전제가 다를 때 많음. |

**한 줄 포지션**: 위 요소들이 **각각** 존재하므로, 우리 차별점은 **label-skew single-global FL에서 diagnostic axes를 함께 측정**하고, **client graph가 scalar diagnostic 또는 graph-free correction을 넘는 정보를 주는지**를 control·baseline으로 검증하는 **diagnostic protocol**에 있다. clustered FL·local drift 보정 계열과의 구분·서지 세부는 **§9.1** 표를 본다.

**FedGH 명칭 주의**: Zhang et al. **IEEE SPL 2024 Gradient Harmonization**과 Yi et al. **header FedGH (arXiv:2303.13137)**는 **다른 계열**이다.

### 5.1 차별성 서술 방식 (문장 가이드)

**차별성 없는 주장(피해야 함)**

- 우리는 처음으로 client graph를 쓴다 / 처음으로 graph smoothing을 FL aggregation에 적용한다.
- 우리는 client graph를 이용한 FL aggregation 방법을 제안한다.
- 우리는 graph smoothing으로 FL 성능을 개선한다(단독 novelty).
- 우리는 gradient conflict를 처음 다룬다.
- 우리는 cancellation/norm reduction을 처음 다룬다.
- 우리는 dominance 문제를 처음 다룬다.

**차별성 있는 주장(현재 방향)**

- 우리는 label-skew single-global FL에서 client interaction을 conflict, cancellation, dominance라는 **diagnostic axes**로 함께 측정한다.
- client graph가 scalar pathology metric과 graph-free correction보다 **추가 정보**를 주는지 controlled ablation으로 검증한다.
- graph smoothing/filtering은 main novelty가 아니라, pathology-aware graph가 유효하다고 확인된 뒤 적용하는 correction operator 후보로 둔다.
- 현재 negative result는 smoothing 전체 실패가 아니라 naive graph construction/operator 조합의 한계를 보여주는 결과로 해석한다.

---

## 6. 현재 negative result의 정확한 의미

이 절에만 아래를 **완전 문장**으로 둔다; 다른 절에서는 “§6과 동일” 수준으로만 언급한다.

### 6.0 독자가 놓치기 쉬운 한 가지

**“그래프가 틀려서” 실패했을 수 있다**는 말은, “우리가 실수로 코드를 잘못 썼다”가 아니라 **엣지를 정의하는 representation이 병리와 정렬되지 않아, 그래프가 통제군(random/shuffled 등)과 구분되지 않는 신호에 가깝다**는 뜻으로 쓴다. 이 경우 smoothing을 아무리 해도 **의미 있는 교정으로 읽히기 어렵다**. 동시에 **operator가 \(\Delta\)를 거의 바꾸지 않는 것**은 별개 가설이므로(§7.4), 둘을 한 문장에 섞어 “그래프가 전부 잘못”이라고 단정하지 않는다.

**핵심 정리 문장**  
현재 결과는 graph smoothing 전체의 실패가 아니라, **(i) 시험한 graph construction이 pathology를 충분히 담지 못했을 가능성**과 **(ii) 제한적인 Laplacian 계열 operator가 \(\Delta\)를 충분히 바꾸지 못했을 가능성**이 겹친 실패에 가깝다. 사전연구 때문에 어려운 것은 graph smoothing 자체가 아니라, 그것을 넓고 단순한 novelty claim으로 내세우는 것이다. 따라서 smoothing/filtering은 폐기 대상이 아니라, **informative graph 후보를 먼저 통제 실험으로 좁힌 뒤** 검증해야 할 correction operator 후보로 남긴다.

### 6.1 무엇이 약했나

- **Phase 2–2.5에서**: `classifier_head_update` 계열 등 **단일 시점 표현**, **코사인 유사도**, **양의 엣지 그래프**, **unnormalized Laplacian smoothing** 조합은 **FedAvgM 대비 안정적인 이점을 주지 못함**.
- **2.5-A**: λ 증가로 **상대 노름 변화량**은 커질 수 있으나 **방향은 거의 동일**(cos≈1), **정확도는 baseline을 넘지 못함** → “λ만 더 키우면 된다”기보다, **당시 graph source가 aggregation pathology를 충분히 담지 못했을 가능성**과 **현재 unnormalized Laplacian smoothing operator가 global update 방향을 충분히 바꾸지 못했을 가능성**을 **함께** 봐야 한다.

### 6.2 무엇으로 일반화하면 안 되나

| 과한 결론 | 맞는 수준의 결론 |
|---|---|
| “graph smoothing 전부 실패” | **해당 naive construction + 해당 operator + 현 설정**에서 실패 |
| “graph-based FL은 의미 없다” | 선행은 **다른 graph source·목적**(collab, topology, multi-round 동적 관계 등) — 우리 결과는 **그 전부의 반증이 아님** |
| “smoothing은 폐기” | **pathology를 담는 graph**가 잡히면 smoothing/filtering은 **여전히 operator 후보** |

### 6.3 smoothing / filtering의 자리

**폐기 대상이 아니다.** 실패한 것은 **instant cosine + positive-cut + unnormalized Laplacian + 현 하이퍼**의 결합이다. 다만 **논문/보고서에서의 novelty 중심**은 smoothing 자체가 아니라, graph가 scalar/graph-free baseline을 넘어 **추가 설명력과 교정 신호를 주는지**를 보여주는 검증 프로토콜로 둔다.

동일 입장 한 덩어리 문장은 **§0 「결론 한 덩어리」**에 둔다.

---

## 7. 다음 실험 계획

### 7.0 최소 1차 ablation (바로 실행 우선)

wishlist처럼 넓히지 말고, **먼저 아래 네 graph source만** 돌린다.

1. **current update graph**  
2. **EMA update graph**  
3. **classifier-head update graph**
   - head state graph는 2차 확장 후보로 둔다.
4. **signed conflict graph**(\(A^+_{ij}=[\cos(g_i,g_j)]_+,\;A^-_{ij}=[-\cos(g_i,g_j)]_+\))

**교정 연산자(1차)**: residual neighbor mixing **또는** normalized Laplacian **중 하나만** 고정해서 graph 간 공정 비교를 맞춘다.

**operator sanity는 §7.4에서 별도로**: unnormalized/normalized/random-walk/signed 계열 비교로 “지금 Δ가 안 바뀌는 게 graph 때문인지 operator 때문인지”를 분리한다.

### 7.1 Graph source ablation (F)

**질문**: 어떤 representation이 **aggregation pathology graph** 후보인가? **§7.0**은 그 최소 실행 단위이다.

필요한 것은 단일 accuracy가 아니라 **아래 세 축이 동시에 찍히는지**다(RQ3).

1. **learned/pathology graph vs** random / shuffled / uniform / identity **control**  
2. **graph 기반 설명·교정 vs** CR/CA/DI/N_eff **scalar diagnostic only** baseline(같은 라운드·같은 예산에서 “스칼라만으로 맞출 수 있는지” 대비)  
여기서 scalar diagnostic only는 correction baseline이 아니라 설명력 baseline으로 둔다. 교정 baseline은 uniform weighting, norm clipping, contribution cap, soft dominance reweighting으로 따로 비교한다.
3. **graph-based correction vs graph-free correction**: uniform weighting, norm clipping, contribution cap, soft dominance reweighting  

이 셋이 없으면 “metric logging + 실패한 smoothing”으로 읽힐 수 있으므로, **차별점은 smoothing 공식이 아니라 이 controlled protocol**에 둔다.

**2차 확장 후보**(1차 결과 본 뒤): accumulated update, dominance-aware influence graph 등.

**필수 control**: shuffled, random, uniform, identity / no-graph.

### 7.2 Graph-free correction과의 직접 비교

§7.1의 **3번**과 동일 배치를 맞춘다. 요약 피벗 규칙은 **§8**을 본다.

### 7.3 Signed conflict graph를 핵심 후보로 승격

positive cosine graph는 conflict 축에서 중요한 음의 관계를 버린다. conflict를 pathology axis로 본다면 signed 처리가 기본 후보가 되어야 한다.

- positive 관계: smoothing/mixing
- negative 관계: attenuation / projection / repulsion 계열
- 비교 대상 operator: signed Laplacian, conflict-aware projection, dominance-aware attenuation

### 7.4 Operator sanity check (graph source와 별도)

Operator sanity check는 graph source가 **완전히** 검증된 뒤에만 할 일이 아니라, **현재 operator가 실제로 global update \(\Delta\)를 바꿀 수 있는지** 확인하기 위해 **별도로 먼저** 수행한다. 다만 **본격적인 성능 개선용 operator tuning**(넓은 λ·아키텍처적 확장)은 **informative graph source**가 확인된 뒤에 단계적으로 넓힌다.

Phase 2.5의 cos≈1은 graph source 문제일 수도, **unnormalized Laplacian 계열이 intervention으로서 \(\Delta\)를 충분히 제어하지 못한 문제**일 수도 있다 — 둘을 분리해서 보려면 sanity가 필요하다.

**확인 항목**

- `cos(Δ_corrected, Δ_base)`
- `||Δ_corrected - Δ_base|| / ||Δ_base||`
- CR / CA / DI / N_eff 변화
- client contribution distribution 변화
- accuracy / stability 변화

**비교할 operator(최소)**

- unnormalized Laplacian(현 재현)·λ sweep  
- normalized Laplacian  
- random-walk Laplacian  
- residual neighbor mixing  
- signed / conflict-aware operator  

### 7.5 Setting sanity (병행)

최소: \(\alpha\in\{0.03,0.1\}\), \(N\ge 10\) 권장. 상세 그리드는 **§9.4**.

---

## 8. Decision rule

아래는 **실험 결과에 따른 피벗**이지, 현재 증명된 사실이 아니다.

### 8.1 Graph smoothing/correction 방향 유지

- learned/pathology graph가 **random/shuffled/uniform/identity control**과 **동시에 비교되어** 우위가 있고,
- **scalar diagnostic**(CR/CA/DI/N_eff) 및 **동일 라운드·동일 budget**의 간단 규칙보다 넘어서는 **추가 정보·교정**을 보이며,
- **graph-free correction**(uniform, norm clipping, cap, soft dominance) 대비도 유리할 때 graph 기반 교정 서사를 유지한다.

### 8.2 Graph claim 낮추고 dominance / data-size weighting pivot

- graph가 **control은 이기나** graph-free correction보다 **일관되게 약하면**, topology 중심 claim은 낮추고 주력 교정 후보를 dominance·data-size 쪽으로 옮긴다.

### 8.3 Diagnostic / failure analysis framework 중심으로 축소

- learned graph가 control을 안정적으로 못 이기지만, scalar diagnostic 로깅이 failure를 **설명력 있게 따라가면**(혹은 regime 구분에 도움이 된다면) 방법 주장보다 **diagnostic protocol**·**failure analysis framework**로 축소한다.

### 8.4 연구 질문 재설계

- graph도 scalar diagnostic도 **설명력이 모두 약하면** RQ·표현·실험 설계를 재정의한다. 이 경우에도 naive graph 한계·control 필수·설정 민감도는 남는 산출물로 정리한다.

---

## 9. 참고문헌 · 상세 근거

### 9.1 가까운 선행연구 매핑 (사실확인용 표)

| 관점 | 가까운 선행연구 | 확인된 내용 | 우리에게 주는 의미 |
|---|---|---|---|
| server-side conflicting **updates/gradients** | **FedGH (Gradient Harmonization)**, Zhang, Sun & Chen, **IEEE SPL** 31: **2595–2599**, **2024** (dblp `journals/spl/ZhangSC24`) | 강한 non-IID에서 **gradient conflict**, 쌍별 **orthogonal plane projection** harmonization, plug-and-play | **Yi et al. FedGH header (arXiv:2303.13137)** 와 혼동 금지. 동일 교정이면 차별 축소 |
| conflict-free 서버 집계 (pFL) | **ConFREE**, Zheng et al., **AAAI 2025** | negative transfer 일부를 업데이트 충돌과 연결, 서버에서 conflict-free 안내 벡터 | 단일 문장 “서버 충돌 보정”만으로는 차별 작음 |
| domain skew fairness, conflict·dominant·bias | **FedHEAL**, Chen et al., **CVPR 2024** | domain skew에서 충돌·지배적 업데이트·aggregation bias | **문제 형식(domain vs label skew)** 구분 필요 |
| dominant gradient averaging bias | **GMA**, Tenison et al., **TMLR** (OpenReview REAyrhRYAo) | 지배적 gradient로 인한 bias·정보 손실, masking 평균 | dominance 각도는 근접 — 우리 가치는 **diagnostic axes + graph control·graph-free baseline 검증 프로토콜** |
| near-orthogonal ↔ global norm reduction | **FedMRUR**, An et al., **NeurIPS 2023** | 직교에 가까운 업데이트와 global norm 축소, 재집계 | cancellation 언어와 직결 — **같은 포인트만 반복하면 novelty 약화** |
| pairwise 유사 + 규모로 collab graph | **pFedGraph**, Ye et al., **ICML 2023** | 모델 유사도·데이터셋 크기로 협력 그래프 | **collaboration** 전형 — 우리는 **pathology 지표·single-global** |
| attentive client 협력 | **FedAMP**, **AAAI 2021** | 개인화 협력 | personalization baseline |
| classifier로 유사 추정 후 집계 | **pFedSim**, arXiv:2305.15706 등 | **로컬 classifier**로 유사 클라이언트 추정 | head 유사도 언어는 겹치나 **목적** 다름 |
| 동적 client relation + attention | **FedAGA**, Ge et al., **Knowl.-Based Syst.** 286 **111399**, **2024** (DOI `10.1016/j.knosys.2024.111399`) | *FedAGA: … enhanced inter-client relationship learning* — 동적 관계·어텐션형 교류 | 단일 라운드 raw 코사인 스냅샷과 다름 |
| GSP·토폴로지 활용 | **G-Fedfilt**, Chen et al. (예: IEEE **IoT J.**, 원고 `arXiv:2212.14395`) | 디바이스 결선 등 **topology** 위 그래프 필터링형 aggregator | graph source가 **행태 업데이트 코사인**과 다름 |
| side info + graph reg | Zhang et al., **Knowl.-Based Syst.** **257:109960**, **2022** (`10.1016/j.knosys.2022.109960`) | 공유 가능 side-information + Laplacian regularization | 그래프 소스 전제 다름 |
| 그래피 데이터 GraphFL | Wang et al., **IEEE ICDM 2022** | 클라이언트가 **그래프 데이터** 보유 | tabular/image label-skew와 문제 클래스 상이 |
| structural entropy 등 | Dai et al., **Inf. Sci.** **718:122338**, **2025** (`10.1016/j.ins.2025.122338`) | 연방 그래프 학습 heterogeneity | 예전 “Structural GFL” 오표기 정정용 |

**이름 혼동**: GraphFL(ICDM 그래피 데이터) vs graph-regularized FL(KBS 2022); FedGH(SPL) vs FedGH(header).

### 9.2 Phase 2.5-A cos / ∥Δ_rel∥ 표

| 표기 | cos(corr,Δ_base) | ∥Δ_rel∥ |
|---|---:|---:|
| A_lap_l0p2_identity | 1.0 | 0.0 |
| A_lap_l0p2_shuffled | 0.999792 | 0.015958 |
| A_lap_l0p2_uniform | 0.999880 | 0.013349 |
| A_lap_l0p2_update | 0.999862 | 0.013183 |
| A_lap_l0p5_identity | 1.0 | 0.0 |
| A_lap_l0p5_shuffled | 0.998788 | **0.039809** |
| A_lap_l0p5_uniform | 0.999125 | 0.037860 |
| A_lap_l0p5_update | 0.999045 | 0.036209 |

### 9.3 Claim boundary

**지금 말해도 되는 것**

- 테스트한 **naive** graph smoothing(즉석 cosine + positive + unnormalized Laplacian)은 **이 설정에서 약했다**.
- Phase 1 수준의 진단 로깅·상관 숫자는 **pilot 근거**일 뿐, seed 3개 correlation을 근거로 물리학처럼 말하면 안 된다.
- dominance/conflict는 **작은 파일럿에서 예비 신호**; graph control **필수**.
- 연구 서사는 **“graph smoothing이 FL을 이긴다”**가 아니라 **pathology 진단 + 유효 representation 검증**으로 좁혀졌다.

**아직 말하면 안 되는 것**

- semantic graph / topology **증명**, spectral filtering **robust**, dominance 교정으로 **정확도 개선 확정**, harmful interaction **최초 제안** 등.

**강한 주장에 필요한 것**: 다수 데이터셋·\(N\)·α·시드·통계·강한 baseline.

### 9.4 왜 naive smoothing이 약할 수 있나 (construction / operator / setting)

**Construction**: instant update 잡음, cosine의 기여도 무시, positive-cut이 negative interaction 제거, **N=5**로 쌍이 10개뿐.

**Operator**: 관찰 대상은 conflict/cancellation/dominance인데 적용은 **positive similarity low-pass**에 가깝다.

**Setting**: α=0.03 극단, rounds=10 짧음, 단일 데이터셋.

**권장 robustness grid**

| α | clients |
|---:|---:|
| 0.03 | 5, 20, 50 |
| 0.1 | 5, 20, 50 |
| 0.3 | 20 |

데이터셋: FashionMNIST + (EMNIST/FEMNIST) + CIFAR-10 등.

### 9.5 부록: venue 감각 (참고)

- **AAAI main**: 아직 method 증거로는 빡센 편; diagnostic이 넓게 일반화되거나 negative-result가 매우 깨끗을 때만.
- **workshop / FL 워크숍**: 현재 스토리로도 **충분히 가능한 층**.
- **analysis 지향 저널**: systematic failure analysis + diagnostic axes + ablation이 맞으면 그 쪽이 현실적.

### 9.6 추적 URL (점검용)

- ConFREE (AAAI 2025): https://ojs.aaai.org/index.php/AAAI/article/view/34449  
- FedGH SPL 2024 (dblp): https://dblp.uni-trier.de/rec/journals/spl/ZhangSC24.html  
- FedHEAL (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Fair_Federated_Learning_under_Domain_Skew_with_Local_Consistency_and_CVPR_2024_paper.html  
- GMA (TMLR): https://openreview.net/forum?id=REAyrhRYAo  
- FedAGA: https://doi.org/10.1016/j.knosys.2024.111399  
- Graph-regularized FL (KBS 2022): https://doi.org/10.1016/j.knosys.2022.109960  
- Dai et al. (Inf. Sci. 2025): https://doi.org/10.1016/j.ins.2025.122338  
- pFedGraph (ICML 2023): https://proceedings.mlr.press/v202/ye23b.html  
- FedMRUR (NeurIPS 2023): https://papers.nips.cc/paper_files/paper/2023/hash/acf2b98eeb09b21968c2de6b1c6952e9-Abstract-Conference.html  
- FedAGHN (KBS 2025): https://www.sciencedirect.com/science/article/pii/S0950705125013942  

### 9.7 빠른 참고: 비교 대상 풀

| 영역 | 예시 |
|---|---|
| FL baselines | FedAvg, FedAvgM, FedProx, SCAFFOLD, FedDyn, FedNova, FedOpt |
| conflict / gradient | PCGrad, FedGH (SPL), ConFREE |
| norm / cancellation | FedMRUR |
| dominance / bias | FedHEAL, GMA |
| personalized graph | FedAMP, pFedGraph, pFedGAT, FedAGHN, pFedLA, pFedSim |
| dynamic relation | FedAGA |
| filtering / GSP / reg | G-Fedfilt, graph-regularized FL (KBS 2022 등) |
| clustered | CFL, IFCA, FedSoft, FedClust류 |
| robust agg | FedMedian, trimmed mean, norm clip, contribution clip |

---

**문서 버전 노트**: 본판은 팀 공유용으로 **중복 서술을 줄이고** §6·§9에 근거·한계를 **한 번 모아** 두었다. 수치·인용 업데이트 시 §2·§9.1–9.2를 먼저 맞추면 본문 흐름은 유지된다.