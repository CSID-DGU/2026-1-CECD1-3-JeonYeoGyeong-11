# Lifecycle Framework Implementation Phases

This directory is now a completed migration archive.  Do not treat it as the
default entry point for new project work.  For day-to-day project execution,
start from the run path in the repository `README.md`, the docs index in
`docs/README.md`, and the current diagnostic configs under `configs/general/diagnostic/`.

## Agent Entry Point

Use the following protocol only when auditing or resuming the historical
Phase 0-7 migration.  It is not the priority path for normal experiment runs or
repository cleanup.

1. Open `CURRENT_STATUS.md` first.
2. Open this directory's `AGENT_IMPLEMENTATION_GUIDE.md`.
3. Open only the phase document named by `CURRENT_STATUS.md`.
4. Follow the phase document's `Agent Task Card` before reading the deeper design sections.
5. Edit only files listed in that phase's allowed scope.
6. Run the tests listed in that phase, then the full test suite.
7. Update `CURRENT_STATUS.md` before reporting completion.

Do not use the design survey as an implementation checklist. It is still normative design reference for support-level classification, lifecycle decomposition, trace vocabulary, and representational coverage.

## Phase Document Contract

Every implementation phase should expose the following agent-facing sections:

- `Agent Task Card`: one-screen summary of objective, scope, tests, and stop conditions.
- `Files to create`: new files the phase may add.
- `Files allowed to modify`: existing files the phase may edit.
- `Files not allowed to modify`: explicit guardrails.
- `Step-by-step implementation order`: the only default execution order.
- `Phase completion report checklist`: required final report structure.

If a phase document and this README disagree, follow the stricter constraint and update the docs before widening scope.

이 디렉터리는 `graph_fl_lifecycle_framework.md`를 실제 코드로 옮기기 위한 단계별 작업 문서다. 목표는 한 번에 큰 refactor를 하지 않고, 각 단계가 독립적으로 테스트되고 되돌아볼 수 있게 만드는 것이다.

구현 전략은 완전한 새 레포 작성이 아니다. 현재 레포 안에 `GraphFLDesign`과 lifecycle core를 거의 greenfield처럼 새로 만들고, 기존 spectral strategy와 graph utilities는 compatibility layer를 통해 점진적으로 옮긴다. 이렇게 해야 이미 있는 Flower runner, data loading, diagnostics, configs, tests를 잃지 않으면서도 새 연구 주제에 맞는 구조를 만들 수 있다.

## 전체 원칙

1. 각 phase는 하나의 구조적 목적만 가진다.
2. phase마다 새 public contract, 변경 파일, 완료 기준, 테스트 기준을 명시한다.
3. 이전 실험 CLI와 기존 결과 포맷은 가능한 한 유지한다.
4. 기존 `graph_source`, `graph_mode`, `aggregation_target`은 바로 제거하지 않는다. 새 lifecycle module 아래의 호환 layer로 옮긴다.
5. actual training path와 shadow diagnostic path를 섞지 않는다.
6. 각 phase가 끝날 때 `unittest discover -s tests`를 통과해야 한다.
7. 코드의 중심 단위는 개별 graph algorithm이 아니라 `GraphFLDesign`이다.

## Framework Claim 고정

이 계획의 최종 목적은 “선행연구 구현체를 모두 모아두기”가 아니다. 핵심 주장은 다음이다.

> graph-FL 방법은 round lifecycle 개입 지점의 조합으로 표현할 수 있다.  
> `GraphFLDesign`과 module contract, 표준 trace, counterfactual diagnostics를 통해 연구자는 기존 설계를 조립/변형하고 새 설계를 빠르게 탐색할 수 있다.

따라서 phase별 결정은 “선행연구 이름을 얼마나 많이 붙였는가”가 아니라, “개입 지점 기반 조립성과 원인 진단 가능성을 얼마나 높였는가”를 기준으로 한다.

## Graph-FL Design Space

Most graph-based FL and personalized graph-FL methods can be decomposed into the following lifecycle decisions. The framework should treat a new graph-FL method as a replacement of one or more lifecycle components, not as a monolithic strategy rewrite.

### 1. Client state

A graph-FL method first decides how to represent each client.

Examples:

- model weights
- model updates
- gradients or pseudo-gradients
- classifier head parameters
- classifier head updates
- EMA updates
- layer-sliced parameters
- local loss or metric vectors
- validation utility vectors
- functional embeddings
- graph-level descriptors
- hash signatures
- mixed moments
- sample-size priors
- hybrid state representations

The framework must not assume that every client state is a single flat update vector. A flat vector is the default core representation, but the lifecycle interface should allow embeddings, signatures, graph descriptors, moment statistics, validation utilities, and hybrid state envelopes.

### 2. Relation estimation

Given client states, a method estimates pairwise or directed relations.

Examples:

- cosine similarity
- Euclidean distance
- RBF similarity
- gradient alignment
- signed gradient conflict
- norm-aware cosine similarity
- validation utility
- Hamming similarity
- DTW distance
- learned attention
- QP-based collaboration score
- graph-autoencoder score
- hybrid relation score

### 3. Topology construction

A relation score is then transformed into a graph topology.

Examples:

- dense weighted graph
- kNN graph
- threshold graph
- directed top-M graph
- cluster graph
- block-uniform graph
- layer-wise graph
- dynamic graph
- learned graph
- identity graph
- uniform graph
- matched random graph
- shuffled graph

### 4. Aggregation use

The graph can be used in different aggregation targets.

Examples:

- update aggregation
- weight aggregation
- gradient proxy aggregation
- graph-smoothed aggregation
- spectral filtering
- cluster-wise aggregation
- personalized row-wise model mixture
- masked model aggregation
- generated personalized weights
- graph-free dominance reweighting

### 5. Extended lifecycle

Modern graph-FL methods may also change the remaining FL lifecycle.

Examples:

- personalized delivery
- previous personalized model delivery
- cluster model delivery
- state across rounds
- local objective regularization
- proximal regularization
- mask regularization
- hypernetwork-generated models

Therefore, the framework should not treat graph construction as a monolithic step. Prior-work-inspired presets may be core-supported, proxy-supported, interface-target, or out-of-scope, but proxy presets must never be described as exact reproductions.

For prior-work mapping details, see [`docs/research/design-pattern-survey.md`](../../graph_fl_design_pattern_survey.md).

## 강제 불변식

아래 항목은 모든 phase에서 유지한다.

1. 모듈 경계: `ClientStateExtractor -> RelationEstimator -> TopologyOperator -> AggregationOperator`의 입력/출력 경계를 흐리지 않는다.
2. 결과 contract: 각 module은 `output + trace`를 반환한다.
3. 조립 단위: 실행 경로 선택의 1급 단위는 `GraphFLDesign`이다.
4. 분기 실험: actual path와 counterfactual path는 같은 round artifact를 공유하되, model update side effect는 actual path에만 허용한다.
5. 실패 명시성: 아직 미구현인 personalized 계열은 조용히 degrade하지 않고 명시적 `NotImplementedError` 또는 지원 수준 표시로 노출한다.
6. 경계 테스트: module 간 import boundary를 테스트로 보호한다.

## 깊은 트리 원칙

phase 진행 중 새 코드는 최대한 lifecycle 하위의 역할별 패키지에 둔다. 하나의 파일에서 여러 lifecycle 단계를 동시에 처리하지 않는다.

```text
spectral_fl/lifecycle/
  contracts/
  contexts/
  modules/
  diagnostics/
  adapters/
```

위 구조는 예시이며 파일명은 달라질 수 있다. 중요한 것은 “단계별 책임 분리”와 “단방향 의존”이다.

## Phase 순서

현재 진행 상태는 `CURRENT_STATUS.md`에 기록한다. 작업을 멈추거나 다시 시작할 때는 이 파일을 먼저 확인한다.

| Phase | 문서 | 목적 |
|---|---|---|
| 0 | `phase_0_repository_simplification.md` | 주제 전환으로 생긴 이전 실험 코드/문서와 현재 framework core를 분리한다. |
| 1 | `phase_1_trace_schema.md` | 모든 모듈이 남길 표준 trace schema를 먼저 만든다. behavior change는 최소화한다. |
| 2 | `phase_2_lifecycle_context_and_contracts.md` | lifecycle context와 module interface를 정의한다. 기존 strategy를 아직 크게 옮기지 않는다. |
| 3 | `phase_3_design_composer_registry.md` | 연구자가 조립하는 설계 단위인 `GraphFLDesign` composer/registry를 만든다. |
| 4 | `phase_4_state_relation_topology_modules.md` | 기존 graph source/builders를 `ClientStateExtractor`, `RelationEstimator`, `TopologyOperator`로 감싼다. |
| 5 | `phase_5_counterfactual_diagnostic_runner.md` | 같은 round artifact로 real/control/cluster/graph-free shadow path를 계산하는 runner를 만든다. |
| 6 | `phase_6_aggregation_delivery_state_hooks.md` | aggregation, delivery, state, local hook 확장 지점을 분리하고, modern graph-FL 방법을 표현하기 위한 core/proxy 실행 경로와 interface-target을 구분한다. |
| 7 | `phase_7_migration_validation_and_docs.md` | 기존 CLI, configs, diagnostics 문서와 테스트를 새 구조 기준으로 정리한다. |

## 성공 기준

이 phase plan이 끝났을 때 프레임워크는 다음 질문에 답할 수 있어야 한다.

- 어떤 client state로 graph를 만들었는가
- relation estimator가 어떤 pairwise score를 만들었는가
- topology 변환이 graph를 얼마나 바꾸었는가
- actual aggregation 전후로 DI, N_eff, alignment, LOO가 어떻게 변했는가
- 같은 round에서 control graph나 graph-free correction이면 내부 지표가 어떻게 달라지는가
- 선행연구형 방법이 exact, proxy, interface 중 어디에 속하는가
- 연구자가 기존 design에서 state/relation/topology/aggregation만 바꿔 새 design을 만들 수 있는가

## 처음부터 새로 만들지 않는 이유

완전한 greenfield 재작성은 개념적으로는 깔끔해 보이지만, 현재 상황에서는 비용이 크다.

| 선택지 | 장점 | 위험 |
|---|---|---|
| 완전 새로 만들기 | 구조를 처음부터 깨끗하게 설계 가능 | Flower 실행, 데이터, 기존 실험, diagnostics, 테스트를 다시 만들어야 함 |
| 현재 코드에 덧붙이기 | 빠르게 기능 추가 가능 | 이전 주제의 구조에 새 주장이 묻힐 수 있음 |
| 새 core + 점진 migration | 새 구조를 깨끗하게 만들면서 기존 자산 재사용 가능 | phase discipline이 필요 |

따라서 선택은 세 번째다.

```text
new core:
  spectral_fl/designs/
  spectral_fl/lifecycle/

existing compatibility:
  spectral_fl/graph/
  spectral_fl/strategies/spectral/
  run_*.py
```

새 core를 먼저 만들고, 기존 코드는 한 번에 제거하지 않는다. strategy는 점점 orchestration만 담당하게 줄여간다.

## 당장 하지 않는 것

다음은 계획에는 남기지만 초기 phase에서 무리하게 구현하지 않는다.

- 모든 선행연구 exact reproduction
- full server GCN 학습
- hypernetwork 기반 client별 model generation
- FED-PUB proxy graph forward와 personalized delivery의 완전 구현
- client local training loop의 대규모 재작성

우선은 현재 spectral strategy와 diagnostic suite가 깨지지 않는 상태에서 lifecycle module contract를 깔고, 그 위로 점진적으로 옮긴다.
