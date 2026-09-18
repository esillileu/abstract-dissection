# F2 Word2Vec 통합 실행 계획

## 진행 기록

| 단계 | 상태 | 완료일 | 검증/결정 |
|---|---|---|---|
| P0 | 완료 | 2026-09-17 | canonical `/f2` URL gate, 전용 test DB gate, read-only DB/S3/MLflow preflight, credential-free identity 출력, immutable run identity 및 새 attempt/predecessor 정책. `just check`: 578 passed, 9 skipped, 7 deselected. |
| P1 | 완료 | 2026-09-17 | normalized WMT/LM1B/UMBC ordered-shard identity 고정, upstream evaluation set commit/SHA-256 고정, W2V1 reconstruction revision과 deterministic minimal slot 생성, unresolved/unverified binding slot 차단. `just check`: 588 passed; `just test-db`: 4 passed. |
| P2 | 완료 | 2026-09-18 | epoch 단위 `TrainingSession`, 완전한 engine state export/restore, schema/config/vocabulary/corpus identity 검증, single-thread bit-identical resume 및 기존 C parity 유지. 커밋 `e3dfd00`; Rust 25 tests passed. |
| P3 | 완료 | 2026-09-18 | PyO3/maturin extension, Python 3.12 workspace 통합, owned NumPy snapshot/restore, Huffman/vocabulary state 직렬화, `Status` 예외 변환, GIL 해제, callback 예외 원자성 및 wheel/editable import 검증. Rust 25 tests, binding 6 tests, package-local `just check` 통과. |
| P4 | 완료 | 2026-09-18 | NEG/HS sampled objective loss, dense observation/epoch report와 Python 노출, 유한값 즉시 검증, append+fsync staging CSV 및 동일 지점 sparse metric 투영. `just check`: 606 passed, 7 deselected. |
| P5 | 완료 | 2026-09-18 | manifest 기반 W2V checkpoint/lookup 직렬화, 전체 file SHA-256·dtype·shape·identity 선검증, manager pointer/retention, MLflow download cache roundtrip, mmap byte-token lookup 및 bit-identical resume. |
| P6 | 완료 | 2026-09-18 | ordered-manifest corpus binding, verified S3 shard cache, exact lexical-token budget streaming, atomic materialization/recovery. F2 non-DB tests: 74 passed. |
| P7 | 완료 | 2026-09-18 | byte-token similarity, 3CosAdd semantic/syntactic analogy, sentence completion scorer, explicit OOV/coverage, deterministic best selection, target/CI report schema. |
| P8 | 완료 | 2026-09-18 | `f2.suites.w2v1` config/spec/executor 등록, local immutable fixture의 2 epoch 실행과 checkpoint resume bit parity, evaluation/check/analysis artifact 검증. F2 W2V targeted 20 tests 통과. |
| P9 | 완료 | 2026-09-18 | F2 전용 tracked runner, canonical corpus materialization, two-attempt epoch resume, remote manifest 검증, durable 이후 catalog link를 구현. 축소 외부 E2E에서 attempt 1 KILLED → attempt 2 FINISHED 및 predecessor/durable 검증. |
| P10 | 완료 | 2026-09-18 | deterministic byte-token phrase materialization/lineage, W2V2 NEG-5/15·HS/full-sentence translation, phrase analogy와 nearest/additive/PCA 평가, 공통 W2V executor 기반 suite, checkpoint resume/mmap lookup 검증. |
| P11 | 완료 | 2026-09-18 | W2V1 24조건×3 seeds와 W2V2 6조건×3 seeds canonical matrix, immutable seed/slot identity, 비용·CPU/thread·승인 정책, target disposition 및 explicit revision/slot/run 보고서 계약. |
| P12 | 완료 | 2026-09-18 | W2V1/W2V2 공통 local/tracked runner, canonical W2V2 tracked lifecycle, 명시적 대규모 실행 승인 gate, corpus shard·phrase pass·epoch·artifact progress, streaming phrase materialization과 전체 smoke 검증. |

P0 구현 메모:

- `repro f2 preflight`는 외부 쓰기 없이 PostgreSQL schema identity, corpus S3
  endpoint와 MLflow API reachability를 확인한다. corpus endpoint는 local 개발의
  loopback HTTP `:9000`/`:19000` 또는 worker의 Tailscale HTTPS `*.ts.net`만 허용한다.
- 운영 DB는 URL 연결 전에 database name이 정확히 `f2`인지 검사한다. 명시적 test
  연결은 `test`, `_test`, `-test` suffix를 가진 전용 database만 허용한다.
- tracked 실행은 `RunIdentity`의 slot/revision/config/resource/manifest identity가
  모두 있어야 시작할 수 있다. resume은 같은 run을 재사용하지 않고 predecessor를
  가진 새 attempt다.
- preflight 결과와 오류에는 credential 또는 전체 URI를 포함하지 않는다.

P1 구현 메모:

- DB의 PASS validation evidence와 shard registry를 read-only audit해 WMT 165개,
  LM1B 80개, UMBC 340개 shard의 ordered-manifest digest와 크기/token metadata를
  catalog manifest에 고정했다.
- `questions-words.txt`와 `questions-phrases.txt`는 upstream commit
  `20c129af10659f7c50e86e3be406df663beff438` 및 SHA-256으로 고정했다.
- canonical `w2v1-reconstruction-r1`은 우선 Table 2 CBOW의 24M lexical-word,
  50 dimension, seed 1, CPU single-thread parity slot만 포함한다. WMT binding은
  unavailable Google News 원자료의 exact 대체가 아니라 `substitute`로 명시한다.
- 모든 required requirement에 verified/checksummed binding이 없는 runnable
  experiment는 planned slot을 가질 수 없도록 manifest validation을 추가했다.

P2 구현 메모:

- 기존 `Trainer::train()`은 별도 전체-epoch loop를 유지하지 않고
  `TrainingSession::train_epoch()`를 반복 호출한다.
- state에는 schema/config/vocabulary/corpus identity, completed epoch, processed token,
  vocabulary state, input/output embedding 및 worker RNG/LR state가 포함된다.
- config/schema/identity mismatch를 별도 `Status`로 거부하며, 단일 스레드의 연속
  학습과 epoch export/restore 후 학습이 bit-identical임을 Rust test로 검증했다.

P3 구현 메모:

- `Corpus`, `VocabularyConfig`, `Vocabulary`, `TrainingConfig`, `Model`, `EpochReport`,
  `TrainingState`, `TrainingSession`을 PyO3로 노출했고 `train_epoch()`는 GIL을 해제한다.
- byte token/offset/count와 input/output embedding은 owned NumPy 배열로 전달하며,
  embedding restore는 float32 C-contiguous shape를 검사한다.
- Python에서 1 epoch state를 export한 뒤 새 model/session으로 복원해 연속 2 epoch
  결과와 동일함을 검증했다.
- workspace 전체 Python 기준을 3.12.14 (`>=3.12,<3.13`)로 올리고 `uv.lock`과 모든
  workspace package 계약을 맞췄다.
- `VocabularyState`의 token/count/Huffman 배열을 Python에서 owned NumPy 배열로
  export/import하며, Rust restore 검증을 재사용해 손상된 offset/path/bit를 거부한다.
- malformed/non-contiguous/non-finite 배열은 mutation 전에 거부하고, epoch 완료 후
  callback 예외가 발생해도 완료된 session state를 export하고 다음 epoch를 계속할 수
  있음을 검증했다. callback의 추가 관측 payload는 P4 관측 계약에서 확장한다.
- workspace dev dependency에 maturin을 고정하고 built wheel 격리 import와 editable
  개발 import를 모두 검증했다.

P4 구현 메모:

- `observation_interval=0`은 관측을 완전히 비활성화하며 기존 C golden embedding
  bits를 유지한다. 양수이면 worker별 model objective invocation 간격으로 NEG/HS의
  binary cross-entropy sum/count를 샘플링한다.
- epoch report와 각 dense observation은 epoch, global processed token, learning rate,
  loss sum/count/mean, elapsed와 throughput을 공유하며 Python callback/report에 owned
  값으로 전달된다.
- objective update 전에 score, loss, gradient와 다음 embedding 값의 유한성을
  검사하고 비정상 progress/rate도 epoch 완료 전에 거부한다.
- F2의 고정-column dense CSV writer는 각 batch를 flush+fsync하며, sparse MLflow
  metric row는 dense point의 token step과 값을 그대로 투영한다. endpoint, path,
  credential을 받을 수 있는 필드는 schema에 없다.

P5 구현 메모:

- Python binding은 complete `TrainingState`를 재구성할 수 있도록 descriptor와 worker
  RNG/LR/objective state를 명시적으로 노출하며 schema와 vocabulary identity를 생성
  시점에 검증한다.
- F2 W2V adapter는 pickle 없이 고정된 NumPy/JSON 파일 집합을 저장하고 manifest의
  전체 file set, SHA-256, dtype, shape, config/vocabulary/corpus/resource identity를
  load 전에 검증한다.
- checkpoint는 `CheckpointManager`의 atomic generation과 `latest`/`best`/`final`
  pointer 의미를 그대로 사용하며 best도 완전한 resumable state만 허용한다.
- lookup artifact는 input embedding과 byte-token index만 포함하고 read-only mmap으로
  token→row→vector 조회를 제공한다. MLflow download cache로 받은 동일 directory도
  같은 검증 경로를 거쳐 복원한다.

P6 구현 메모:

- F2 W2V corpus binding은 resource version과 ordered shard index/URI/SHA-256/byte
  size/word/document count를 하나의 canonical manifest digest로 고정하며, 누락·재정렬된
  shard 또는 digest가 다른 binding은 다운로드 전에 거부한다.
- F2가 환경에서 구성한 read-only object store를 adapter에 주입하고, adapter는
  `repro-io`의 `get_file` 계약으로만 object bytes를 읽는다. shard cache와 임시
  materialization은 각각 `RuntimePaths.cache_root`와 `staging_root` 아래에만 둔다.
- 각 cached shard는 사용 전 size와 SHA-256을 다시 확인한다. partial/corrupt download는
  폐기 후 재시도하며, 완성된 corpus와 identity sidecar는 fsync 뒤 atomic rename으로
  게시한다. 손상된 결과 cache는 검증된 shard cache에서 재구성한다.
- lexical token budget은 manifest 순서로 스트리밍하며, budget이 record 중간에 오면
  해당 token까지만 쓰고 newline으로 닫는다. 따라서 budget을 넘기지 않으며 같은
  binding/budget은 cache hit, retry, 재구성 여부와 무관하게 같은 bytes와 identity를 낸다.

P9 구현 메모:

- `uv run repro f2 preflight`로 canonical PostgreSQL, corpus S3, MLflow의 read-only
  접근을 확인했다. 출력은 credential과 전체 URI를 포함하지 않았다.
- `uv run repro run f2 w2v1 -e 01 -a d50-w24m --tracking-uri ... --dry-run`은
  canonical slot 하나만 선택함을 확인했다.
- F2 CLI는 W2V1 전용 tracked runner를 주입한다. runner는 catalog/DB의 verified ordered
  shard binding을 S3에서 materialize하고 실제 corpus SHA-256을 executor에 전달한다.
- 첫 attempt는 1 epoch checkpoint를 게시하고 KILLED/interrupted로 보존한다. 두 번째
  attempt는 MLflow artifact cache로 그 checkpoint를 다시 내려받아 복원하며 predecessor
  run ID를 기록한다. 전체 artifact manifest의 원격 SHA-256 검증이 끝난 뒤에만 durable
  complete를 설정하고 canonical slot을 catalog에 연결한다.
- 외부 축소 E2E는 `local-smoke`로 실행했다. attempt 1 `da0bcad4…`는 KILLED/비내구,
  attempt 2 `229c947c…`는 FINISHED/durable이며 predecessor lineage와 remote manifest를
  재검증했다. smoke run은 canonical catalog slot에 연결하지 않는다.
- full 24M 사전 점검에서 catalog에 고정한 digest가 DB verified 165-shard digest와 달라
  실제 ordered digest로 교정했다. 또한 fixture용 hash capacity와 observation cadence가
  canonical에 상속되던 문제를 분리했다. 비용이 큰 full canonical 학습은 이 단계의
  외부 lifecycle 검증에 사용하지 않았다.

P10 구현 메모:

- classic word2phrase score를 문서 경계별 왼쪽 우선으로 적용하며 pass 수, threshold,
  min count와 byte separator를 policy digest 및 derived-corpus lineage에 고정한다.
- `w2v2`는 공통 checkpoint/lookup/evaluation과 W2V1 epoch executor를 재사용하고,
  phrase materialization 정책만 실행 전에 합성한다.
- NEG-5/NEG-15/HS와 whole-sentence window 변형을 직접 engine config로 번역한다.
  미지원 NCE는 NEG로 대체하지 않고 config validation에서 명시적으로 거부한다.
- phrase analogy는 기존 byte-token 3CosAdd 계약을 사용하며 nearest entity,
  additive composition 및 sign-stable PCA projection을 공통 평가 모듈에 추가했다.

P11 구현 메모:

- canonical catalog은 W2V1 Table 2의 24개 dimension/token 조건과 W2V2 phrase
  objective/subsampling 6개 조건을 seeds 1/7/19로 확장한 90개 slot을 고정한다.
- 각 slot은 예상 token update, CPU 단일 thread, 대규모 실행 전 명시적 승인 필요 여부를
  기록한다. plan materialization이나 검증 과정은 실제 대규모 학습을 시작하지 않는다.
- 원 자원 부재는 reconstruction으로 명시하고, 비교 전용 external baseline과 현재
  unsupported target을 별도 disposition으로 유지한다. exact reproduction 분류도 계약에
  포함하되 현재 runnable matrix에는 원 corpus 부재로 해당 slot이 없다.
- planner-selected seed가 runtime config, planned slot, MLflow lineage에 동일하게 반영되며
  plan revision은 execution plan identity에서 파생한다.
- 분석 보고서는 explicit plan revision/planned slot/MLflow run ID를 요구하며 target CI,
  coverage, throughput 및 hardware identity를 Markdown/CSV로 출력한다.

P12 구현 메모:

- W2V1에 묶여 있던 tracked lifecycle을 공통 W2V 책임으로 이동하고 W2V2 canonical
  실행도 동일한 preflight, corpus binding, attempt resume, durable artifact, catalog link를
  사용한다.
- local fixture도 공통 Runner를 사용하므로 run/epoch progress가 표시되며 seed별 staging
  identity가 분리된다.
- canonical 학습은 `--approve-large-run`을 명시해야만 시작한다. dry-run과 local smoke는
  승인이 필요 없다.
- corpus shard/token, phrase pass/document, epoch token/loss/throughput, artifact publish가
  progress output에 나타난다. phrase materialization은 전체 corpus를 메모리에 적재하지
  않고 두 번의 streaming scan으로 처리한다.
- local W2V1/W2V2 CLI smoke, smoke marker 12개, F2 non-external 92개, DB 4개, network
  3개, external read-only preflight와 root `just check` 633개를 모두 통과했다.

## 1. 목표와 완료 상태

이 계획은 다음 두 문서를 하나의 실행 순서로 통합한다.

- `F2_W2V_EXPERIMENT_READINESS.md`: F2 연구, corpus, catalog, suite, 운영 준비
- `W2V_PYTHON_BINDING_PLAN.md`: Rust epoch API, Python binding, checkpoint와 관측 계약

목표는 `w2v1`과 `w2v2`를 다음 단일 흐름으로 재현 가능하게 만드는 것이다.

```text
catalog의 canonical plan 선택
→ immutable corpus binding과 shard 검증
→ Rust vocabulary/model/session 구성
→ epoch 학습·관측·평가
→ checkpoint를 MLflow에 내구적으로 게시
→ epoch 경계에서 중단·재개
→ lookup embedding 게시
→ 결과 다운로드·분석
→ planned slot과 MLflow run 연결
```

최종 완료 조건은 축소 smoke뿐 아니라 canonical plan을 안전하게 생성하고 실행할
수 있는 상태다. 전체 논문 matrix 실행 자체는 별도 비용 집행 단계이며, 이 계획은
그 실행을 시작할 수 있는 검증된 시스템과 확정된 plan revision까지를 범위로 한다.

## 2. 고정 책임 경계

| 계층 | 책임 | 금지 사항 |
|---|---|---|
| Rust `packages/w2v` | vocabulary, model, objective, worker, epoch session, 복원 가능한 수치 상태, 구조화된 관측 | NumPy 파일, MLflow, F2 이름·경로·정책 의존 |
| Python extension | Rust/Python 타입 변환, NumPy snapshot/restore, 예외 변환, GIL 관리 | checkpoint/MLflow/실험 정책 소유 |
| F2 adapter | spec/config 변환, corpus materialization, checkpoint 및 embedding 직렬화, 평가 입력 변환 | 범용 실행 lifecycle 재구현 |
| F2 executor/suite | epoch loop, 평가 주기, progress, run 상태와 artifact 정책 | 수치 학습 구현 |
| `repro-core` | 실행 정의, runtime path, checkpoint generation/retention/pointer | W2V 상태 형식 또는 MLflow 의존 |
| `repro-mlflow` | metric/artifact 게시, durable 검증, checkpoint download cache | F2 연구 정책 |
| F2 catalog/DB | 논문 target, resource substitution, plan revision, planned slot, run lineage | corpus bytes나 run artifact 저장 |

추가 불변 조건:

- resume 정밀도는 **완료된 epoch 경계**다. token 중간 cursor 저장은 범위 밖이다.
- 단일 스레드 resume는 bit-identical을 목표로 한다. 병렬 실행은 유한값과 통계적
  일관성을 검증하되 bit-identical 순서를 약속하지 않는다.
- byte token을 UTF-8로 강제 변환하지 않는다.
- 학습 중 mutable zero-copy 배열을 Python에 노출하지 않는다. snapshot은 epoch
  경계에서만 만든다.
- checkpoint는 완전한 재개 상태, lookup artifact는 소비용 input embedding이다.
- MLflow는 run/checkpoint의 SSOT, S3는 corpus bytes의 SSOT, PostgreSQL은 F2
  metadata의 SSOT다. `.staging`과 `.cache`는 각각 휘발성·재구성 가능 영역이다.
- 실제 endpoint와 credential은 코드, log, artifact, committed config에 넣지 않는다.

## 3. 현재 기준선

이미 존재하므로 다시 만들지 않을 기반:

- Rust CBOW/Skip-gram, HS/NEG, subsampling, LR schedule, single/multi-thread,
  vocabulary/Huffman/negative table, embedding snapshot, C oracle parity
- Rust epoch session과 schema/identity가 있는 완전한 state export/restore,
  single-thread resume parity
- PyO3/maturin Python extension의 핵심 객체, owned NumPy snapshot/restore,
  GIL을 해제하는 epoch 학습과 Python resume parity
- Python 3.12.14 기반 workspace와 재생성된 `uv.lock`
- F2 PostgreSQL `catalog`/`corpus`, suite registry와 `ExecutionDefinition` builder
- `repro-core` checkpoint generation/retention/pointer와 `RuntimePaths`
- `repro-mlflow` artifact/metric 게시 및 download cache
- normalized LM1B/UMBC/WMT corpus와 DB lineage/validation metadata
- `studies/f2/catalog/w2v.json`의 paper, target, experiment spec, resource,
  requirement 기록

현재 남은 핵심 공백:

- 구현·스모크 기준의 공백은 없다. 남은 작업은 승인할 canonical slot을 선택해 실제
  비용이 드는 학습 matrix를 실행하고 결과를 분석하는 운영 단계다.

## 4. 의존성 그래프

```text
P0 기준선·운영 안전성 ───────────────┬───────────────┐
                                     │               │
P1 연구 계약·resource binding ──────┤               │
                                     │               │
P2 Rust epoch/state API ──→ P3 Python binding       │
          │                         │                │
          └────────→ P4 관측 ──────┤                │
                                    ├→ P5 artifact/checkpoint
P1 ─→ P6 corpus adapter ────────────┤                │
P1 ─→ P7 evaluation ────────────────┤                │
                                    ↓                │
                         P8 w2v1 vertical slice ←────┘
                                    ↓
                         P9 tracked resume E2E
                                    ↓
                         P10 w2v2 phrase 확장
                                    ↓
                         P11 canonical matrix/reports
```

P2, P6, P7은 P0/P1의 계약이 고정된 뒤 병렬로 진행할 수 있다. P3은 P2의 공개 API,
P5는 P2/P3의 상태 표현에 의존한다. P8 이전에는 외부 서비스를 쓰지 않는 local
fixture 검증을 우선하며, P9부터 P0의 외부 서비스 preflight를 반드시 통과해야 한다.

## 5. 단계별 작업

### P0. 기준선 확정과 운영 안전성

**목적:** 구현 도중 잘못된 DB나 endpoint에 쓰지 않도록 실행 경계를 먼저 고정한다.

작업:

- 현재 `w2v.json`, corpus resource/version, Rust public API, checkpoint 및 MLflow
  계약을 inventory로 고정하고 이미 완료된 항목을 중복 구현하지 않는다.
- 모든 F2 process가 `F2_DATABASE_URL`의 canonical `/f2`만 사용하게 preflight하고
  `/f2_db`, subsystem alias, localhost fallback이 없음을 확인한다.
- DB integration test는 `F2_TEST_DATABASE_URL` 또는 disposable PostgreSQL 18만
  사용하며 application/production dotenv로 fallback하지 않게 검증한다.
- production run 허용 조건과 lifecycle을 정한다. 최소한 planned slot이 없거나
  resource/config digest가 다르면 tracked run을 시작하지 않는다.
- `F2_CORPUS_S3_ENDPOINT`가 실제 Tailscale HTTPS endpoint인지 worker에서
  read-only preflight한다. secret redaction test를 둔다.
- Common Crawl 분석은 명시적 run ID만 허용하고 legacy DB는 실행 fallback으로
  사용하지 않는다. archive/read-only 전환은 repository 외 운영 handoff로 기록한다.
- F2 MLflow experiment naming과 resume 의미를 결정한다. 권장 정책은 각 실행
  attempt를 별도 MLflow run으로 만들고 parent planned slot 및 predecessor run ID로
  이어 붙이는 것이다. 동일 run 재개보다 실패/재시도 lineage가 명확하다.

산출물/게이트:

- 외부 쓰기 없이 DB/S3/MLflow 설정과 대상 identity를 검사하는 preflight
- test가 production/shared service를 가리키면 즉시 실패하는 검증
- credential/URI redaction test
- 이후 단계가 사용할 run identity/tag/status 문서화

### P1. 연구 계약과 canonical resource binding

**의존:** P0

**목적:** 코드가 임의로 corpus, seed, 평가 metric을 선택하지 않게 catalog를
실행 가능한 계약으로 완성한다.

작업:

- 기존 paper/target/spec/requirement를 audit하고 논문 원조건과 reconstruction을
  분리한다. unavailable 원자료는 그대로 unavailable로 남기고 substitute를 별도
  resource binding으로 기록한다.
- `w2v1`부터 normalized WMT/LM1B/UMBC의 정확한 resource version, shard set/order,
  manifest digest, token budget, 조기 종료 규칙을 결정한다.
- 원 corpus 대비 domain/time/size 차이와 exact/reconstructed provenance status를
  기록한다.
- seed set, 반복 수, CPU/thread/device 정책을 정한다. single-thread parity seed와
  production throughput seed/mode를 구분한다.
- word analogy, similarity, MSR sentence completion, phrase analogy dataset의
  immutable resource version과 license/access 상태를 등록한다.
- metric 이름, 방향, OOV 정책, coverage, CI/오차 비교 방법과 best checkpoint
  metric을 확정한다.
- `w2v1` 최소 vertical slice용 binding과 slot을 먼저 만들고, 검증 후 전체
  `w2v1`, 마지막으로 `w2v2` canonical plan revision을 생성한다.

산출물/게이트:

- non-empty `resource_bindings`
- revisioned canonical plan과 deterministic `planned_run_slots`
- 동일 manifest 입력의 materialization idempotency test
- unresolved required resource가 있는 slot은 생성/실행되지 않는 검증

### P2. Rust epoch session과 완전한 상태 계약

**의존:** P0, P1의 seed/config/digest 계약

**목적:** 기존 학습 구현을 중복하지 않고 `Trainer::train()`의 epoch loop를
`TrainingSession`으로 직접 재구성한다. 기존 전체 학습 경로는 새 session을
호출하게 바꾸고 obsolete loop는 제거한다.

작업:

- `train_epoch()`, `completed_epochs()`, `is_complete()`와 `EpochReport`를 추가한다.
- completed epoch, processed/global tokens, 다음 LR progress를 session 상태로 둔다.
- root seed, worker ID, epoch, purpose에서 worker RNG를 결정적으로 파생한다.
- input/output embedding restore를 구현하고 shape, dtype-converted values,
  finite values, vocab size, dimension을 검증한다. 실행 중 restore는 거부한다.
- token bytes/offsets, counts, retained order, Huffman offsets/paths/bits 및
  deterministic hash rebuild에 충분한 vocabulary export/restore를 구현한다.
- schema version, config digest, vocabulary digest, corpus resource/version digest를
  포함한 engine-neutral state descriptor를 노출한다.
- config/vocabulary/corpus mismatch와 지원하지 않는 schema version을 명시적
  `Status`로 거부한다.

검증 게이트:

- 한 epoch session 결과가 기존 `train()` 결과와 동일
- N epoch 연속 실행과 K epoch export/restore 후 실행이 single-thread bit-identical
- vocabulary row ordering, Huffman state, input/output embedding roundtrip 동일
- malformed shape/digest/schema/non-finite restore 거부
- 기존 C oracle parity 유지

### P3. PyO3/maturin Python extension

**의존:** P2

**목적:** Rust 상태와 학습 호출을 얇고 명시적인 Python API로 노출한다.

작업:

- `packages/w2v`에 PyO3/maturin scaffold와 workspace에서 소비 가능한 Python
  package를 구성한다. wheel과 editable build를 검증한다.
- `Corpus`, `Vocabulary`, `Model`, `TrainingSession`, config enum/dataclass 변환을
  제공한다.
- 긴 `train_epoch()` 동안 GIL을 해제한다.
- epoch 종료 snapshot으로 input/output embeddings와 byte token/offset/count/
  Huffman 배열을 NumPy로 export/import한다.
- 기본값은 안전한 owned copy로 한다. read-only view가 실제 이득이 있고 lifetime을
  증명할 수 있을 때만 별도 API로 추가한다.
- Rust `Status`를 구체적인 Python exception으로 변환한다.
- callback은 제한된 관측 경계에서만 GIL을 재획득하고, Python 예외 발생 시
  session을 손상시키지 않은 채 epoch 실패를 전달한다.

검증 게이트:

- NumPy dtype/shape/contiguity 및 byte token roundtrip
- 잘못된 배열과 callback exception의 원자적 실패
- 학습 중 mutable view 부재와 GIL release 동작
- built wheel import 및 editable 개발 import

### P4. 관측, progress, dense metric 계약

**의존:** P2; Python 전달은 P3

**목적:** Rust는 측정값만 만들고 Python이 표시·저장 정책을 소유하게 한다.

작업:

- NEG와 HS objective에 loss sum/count 관측을 추가한다.
- hot path 비용을 제어하는 observation sampling interval을 config에 둔다. 이는
  metric 저장 간격과 별개다.
- event/report에 epoch, processed/global tokens, LR, sampled objective loss,
  elapsed, tokens/sec, observation count를 포함한다.
- NaN/Inf loss/embedding과 비정상 progress를 즉시 실패시킨다.
- Python progress bar와 dense observation schema를 정의한다. dense series는
  `.staging`에 append/flush하고 종료 후 artifact로 게시한다.
- MLflow에는 epoch 또는 큰 token interval의 sparse metric만 기록하며 dense
  artifact와 step/epoch 의미가 일치하게 한다.

검증 게이트:

- 관측 비활성화 시 기존 수치 결과 불변
- sampling interval별 count/loss 집계 정확성
- sparse metric과 dense artifact의 공통 지점 값 일치
- 중단 직전 flush까지 복구 가능하고 secret/path가 기록되지 않음

### P5. W2V checkpoint와 lookup artifact

**의존:** P2, P3; manager 연동은 P0의 run 정책

**목적:** generic pickle adapter를 확장하지 않고 W2V의 명시적, 검증 가능한
직렬화 adapter를 만든다.

재개 checkpoint:

```text
checkpoint/
├── input_embeddings.npy
├── output_embeddings.npy
├── token_bytes.npy
├── token_offsets.npy
├── counts.npy
├── huffman_offsets.npy
├── huffman_paths.npy
├── huffman_bits.npy
├── trainer_state.json
└── manifest.json
```

lookup artifact:

```text
embeddings/
├── input_embeddings.npy
├── token_bytes.npy
├── token_offsets.npy
├── counts.npy
└── manifest.json
```

작업:

- manifest에 format version, file SHA-256, dtype/shape, config/vocab/corpus digest,
  resource version, completed epoch/global tokens를 기록한다.
- load 전에 전체 manifest와 digest를 검증하고, staging에서 완성한 뒤
  `CheckpointManager` generation으로 atomic publish한다.
- `latest`, `best`, `final`, optional periodic retention을 기존 manager 의미에 맞춘다.
- MLflow upload 후 remote manifest를 검증한 경우에만 durable complete로 표시한다.
- resume은 `repro-mlflow` download cache를 통해 내려받은 동일 artifact로 검증한다.
- lookup artifact는 float32 `[vocab_size, dimension]` input embedding과 byte token
  index만 포함하며 `np.load(..., mmap_mode="r")`를 지원한다.

검증 게이트:

- checkpoint save/load SHA-256 및 session resume parity
- corrupt/missing file과 identity mismatch를 게시/복원 전에 거부
- `latest`/`best`/`final` pointer와 retention 검증
- MLflow roundtrip 후 동일 manifest/digest
- mmap token→row→vector lookup 검증

### P6. Corpus streaming/cache adapter

**의존:** P0, P1

**목적:** catalog binding을 실제 immutable local input으로 변환한다.

작업:

- F2가 S3 환경을 해석해 explicit `repro-io` config로 전달하고, object bytes는
  `repro-io`를 통해 읽는다.
- `RuntimePaths.from_environment()`의 cache/staging 경로만 사용한다.
- manifest 순서대로 shard를 fetch/cache하고 object digest, size, manifest digest를
  검증한다. cache는 검증 실패 시 사용하지 않는다.
- token budget과 record/shard 경계의 조기 종료 의미를 명시하고 deterministic하게
  적용한다.
- 우선 Python adapter가 `.zst`를 검증·materialize하여 Rust에 local corpus path를
  넘긴다. Rust direct-zstd streaming은 profiling으로 I/O 병목이 확인될 때만 별도
  변경으로 고려한다.
- cache hit, partial download, retry 뒤에도 동일 local manifest identity를 만든다.

검증 게이트:

- fixture S3/local objects를 이용한 order/digest/budget test
- corrupt/truncated/reordered shard 거부
- cache 재구성과 interrupted materialization recovery
- worker endpoint read-only preflight; 실제 corpus를 test fixture로 사용하지 않음

### P7. Epoch 평가와 분석 계약

**의존:** P1; embedding 소비는 P3/P5

**목적:** best 선택과 논문 비교가 실행 후 임의 분석이 아니라 사전 명세를 따르게
한다.

작업:

- 공통 byte-token lookup과 word similarity, semantic/syntactic analogy scorer를
  구현한다.
- OOV 질문 제외/실패 정책, 전체/유효 question 수와 vocabulary coverage를 항상
  함께 기록한다.
- 기본은 epoch마다 평가한다. 비용이 큰 경우 동일 hook에서 configured interval
  또는 catalog에 명시된 subset을 사용한다.
- target별 paper value, reconstruction estimate, CI/오차 비교를 분석 schema로
  정의한다.
- `w2v1` word analogy/sentence completion을 먼저 지원한다.
- phrase detection/training representation과 phrase analogy는 P10에서 확장한다.

검증 게이트:

- 작은 고정 vocabulary/vector fixture의 exact scorer 결과
- OOV/coverage/semantic/syntactic split 집계
- best metric direction 및 tie-breaker deterministic
- 동일 run set에서 동일 report/CI 생성

### P8. `w2v1` 최소 vertical slice

**의존:** P1~P7의 최소 기능

**목적:** 추상화를 더 늘리기 전에 한 조건·한 seed의 local end-to-end를 완성한다.

작업:

- `f2.suites.w2v1`에 config, spec, adapters, executor, evaluation, analysis를 만든다.
- `SuiteCatalog`로 module-scoped `ExecutionDefinition`을 만들고 `F2Definition` 및
  `plan/run/analyze/check` CLI에 등록한다.
- executor가 corpus adapter → Rust binding → epoch loop → evaluation → checkpoint
  → lookup artifact 순서를 소유하게 한다.
- 먼저 작은 local immutable fixtures로 1 epoch smoke와 2 epoch parity를 수행한다.
- catalog minimal slot의 identity와 실제 resolved config digest가 일치해야 실행한다.

게이트:

- 한 명령으로 local 1-epoch smoke 완료
- 2 epoch 연속 실행과 1 epoch + checkpoint resume의 single-thread parity
- analysis report와 `check`가 누락 artifact/metric을 탐지
- suite 외부에 임시 W2V 전용 실행 경로를 남기지 않음

### P9. Tracked resume vertical slice

**의존:** P0 preflight, P8

**목적:** 실제 외부 service 경계와 durable lifecycle을 최소 비용으로 검증한다.

작업:

- worker S3/DB/MLflow preflight 후 최소 canonical reconstruction slot 하나를 실행한다.
- run tag에 paper, suite, experiment spec, variant, seed, config digest, corpus resource
  version/manifest digest, planned slot ID를 기록한다.
- checkpoint, dense metrics, final embedding 경로를 고정한다.
- 의도적으로 epoch 경계에서 중단한 뒤 새 attempt run으로 resume하고 predecessor
  lineage를 기록한다.
- durable completion 검증 후에만 catalog slot에 final MLflow run ID를 연결한다.
- failed/interrupted attempt는 보존하되 complete slot으로 계산하지 않는다.

게이트:

- remote checkpoint download/cache resume 성공
- remote artifact SHA-256과 local manifest 일치
- sparse/dense metric 및 catalog/MLflow/corpus lineage 일치
- run ID를 명시한 분석 report 생성
- secret과 credential-bearing endpoint가 log/artifact에 없음

### P10. `w2v2` phrase 확장

**의존:** P9, P1의 phrase resource 결정

**목적:** `w2v1` 경로를 복제하지 않고 실제로 다른 phrase/objective/evaluation
정책만 확장한다.

작업:

- phrase detection 횟수, threshold, min count, joining/token byte representation을
  catalog spec과 corpus lineage로 고정한다.
- phrase corpus artifact/resource version을 생성·검증하는 F2 adapter를 추가한다.
- NEG-5/NEG-15/HS, subsampling, full-sentence context 등 target matrix를 Rust config로
  직접 번역한다. NCE처럼 Rust가 지원하지 않는 objective는 구현하거나 해당 target을
  unsupported/external baseline으로 명시하며 NEG로 대체하지 않는다.
- phrase analogy, nearest entity, additive composition, PCA/qualitative report를 추가한다.
- `f2.suites.w2v2`는 공통 W2V adapters/executor mechanism을 재사용하고 phrase 정책만
  소유한다.

게이트:

- phrase corpus 생성의 deterministic lineage/digest
- phrase vocabulary checkpoint/resume와 mmap lookup
- phrase analogy OOV/coverage 및 paper target report
- `w2v1` 회귀 없음

### P11. Canonical matrix 확장과 최종 readiness

**의존:** P9, P10

작업:

- 검증된 template로 `w2v1`, `w2v2`의 canonical plan revision과 seed 반복을
  materialize한다.
- exact reproduction, reconstruction, external baseline, unsupported target을 matrix에서
  명확히 구분한다.
- 비용·예상 token 수·device/thread 정책을 plan 단계에서 표시하고 승인되지 않은
  대규모 slot은 실행하지 않는다.
- run selection은 explicit plan revision/slot/run ID를 사용한다.
- target별 CI/오차, coverage, throughput과 hardware-normalized 해석을 human-readable
  `artifacts/analysis/f2/<suite>/` 보고서로 만든다.
- obsolete 임시 config, duplicate adapter, old training loop와 compatibility wrapper를
  제거한다.

최종 게이트:

- 모든 planned slot이 immutable resource/config/seed identity를 가짐
- plan → run → interrupt/resume → durable artifacts → analyze → catalog link E2E
- checkpoint/embedding digest, mmap lookup, sparse/dense metric consistency
- suite별 unit/integration test와 repository architecture checks 통과

## 6. 검증 전략과 명령 원칙

각 단계는 가장 작은 범위의 test를 먼저 실행한다. Python 관련 명령은 항상
`uv run`을 사용한다. 실제 DB integration과 network smoke는 기본 gate와 분리하며,
전용 test DB 또는 명시적 read-only endpoint가 없으면 실행하지 않는다.

`packages/w2v/`를 수정한 단계에서는 그 디렉터리의 `AGENTS.md`가 우선한다.
따라서 package-local `just check`를 실행하며 root `just check`나 monorepo-wide
pytest를 그 작업의 검증으로 실행하지 않는다. F2/Python 통합 단계에서는 관련
targeted test를 거친 뒤 repository 지침의 root `just check`를 실행한다. 환경 또는
외부 service 문제로 gate가 시작/완료되지 않으면 환경을 수리하지 말고 즉시 중단해
실패 원인과 실행되지 않은 check를 보고한다.

필수 자동 검증 묶음:

1. Rust: 기존 C parity, epoch equivalence, single-thread resume, restore rejection,
   observation overhead/accuracy.
2. Binding: wheel/import, NumPy roundtrip, byte token, exception/GIL/lifetime.
3. Adapter: corpus order/digest/budget, checkpoint manifest/corruption, mmap lookup.
4. Evaluation: exact fixture scores, OOV/coverage, best selection, reproducible reports.
5. Suite: plan identity, local E2E, CLI discovery, architecture dependency boundaries.
6. External smoke: dedicated preflight, S3 read, MLflow durable roundtrip, catalog slot link.

## 7. 명시적 비범위와 보류 결정

- token 중간 checkpoint와 worker cursor 복원
- parallel bit-identical resume 보장
- Rust 내부의 NumPy/MLflow/S3/F2 정책
- repository가 외부 PostgreSQL, S3, MLflow를 배포·관리·백업하는 기능
- unavailable original corpus를 substitute와 동일하다고 취급하는 것
- profiling 근거 없는 Rust direct-zstd reader, 범용 plugin/factory/새 checkpoint framework
- 검증 전 전체 수십억 token matrix 실행

아래 결정은 해당 단계 진입 전에 catalog 또는 명시적 계약으로 닫아야 한다.

| 결정 | 마감 단계 | 기본 방향 |
|---|---|---|
| W2V1 substitute corpus와 token budgets | P1 | immutable normalized resource binding |
| seed/repetition/thread/device | P1 | parity와 throughput 모드를 분리 |
| best metric과 OOV 정책 | P1/P7 | coverage 동반, 방향/tie-break 명시 |
| resume run lifecycle | P0 | 새 attempt run + predecessor lineage |
| loss observation cadence | P4 | sampled observation, sparse MLflow |
| zstd 처리 위치 | P6 | Python materialization 우선 |
| phrase construction | P10 | versioned derived corpus lineage |
| unsupported NCE 등 objective | P10 | 명시적 unsupported 또는 실제 구현; 대체 금지 |

## 8. 실질적 마일스톤

- **M1 — Engine resumable:** P2 완료. Rust만으로 epoch parity와 restore 가능.
- **M2 — Python consumable:** P3~P5 완료. NumPy/checkpoint/lookup roundtrip 가능.
- **M3 — Local W2V1 ready:** P6~P8 완료. 외부 write 없이 vertical slice 통과.
- **M4 — Tracked W2V1 ready:** P9 완료. 실제 durable resume와 lineage 통과.
- **M5 — W2V2 ready:** P10 완료. phrase pipeline과 평가 통과.
- **M6 — Experiment ready:** P11 완료. canonical revisions/slots와 최종 gate 통과.

각 마일스톤은 앞 단계의 검증을 통과하기 전 다음 단계의 대규모 실행을 허용하지
않는다. 특히 M3 이전에는 production corpus/run write를, M4 이전에는 전체 `w2v1`
matrix를, M5 이전에는 전체 `w2v2` matrix를 시작하지 않는다.
