
# F2 실행·저장 아키텍처 리팩토링 설계안

> **상태:** 구현 완료 (v1.1, 2026-11-03)  
> **기준:** `esillileu/abstract-dissection`, `ref/runner`  
> **대상:** F2 실행·저장·분석 체계, 관련 공통 패키지의 경계  
> **비대상:** DLFS 실행 체계 개편, F2-CC의 코퍼스 생산 파이프라인 전면 개편  
> **작업 범위:** F2 단일 실행 생명주기 및 저장·게시·분석 계층 통합 구현 완료. DLFS와의 격리 유지.

## 0. 핵심 결정

1. **재사용성 우선.** 새 논문을 추가할 때 논문별로 다시 작성하는 것은 실험 명세, 고유한 자원 준비, 특수 계산, 논문별 분석뿐이다. 실행 여부 판단·로컬 저장·MLflow 게시·결과 선택·캐시·공통 평가 방식은 재사용한다.
2. **F2의 실행 체계를 통일하되 DLFS와는 통합하지 않는다.** F2에는 하나의 실행 생명주기를 적용한다. DLFS는 현행 레거시 실행 체계로 남긴다. F2 때문에 DLFS에 새로운 인터페이스를 강제하지 않는다.
3. **하위 호환성은 목표가 아니다.** 이전 API, 저장 형식, 파일 경로를 위한 래퍼·별칭·fallback을 추가하지 않는다. 필요한 데이터는 일회성 마이그레이션하고, 필요 없는 기능은 삭제하거나 레거시로 격리한다.
4. **공통화의 기준은 실제 구현의 공유 여부다.** `packages` → F2 공통 → 알고리즘군 공통 → 논문별 순으로 책임을 내려보낸다. 외형만 맞추기 위해 빈 모듈·추상 클래스·프로토콜을 만들지 않는다.
5. **실행과 저장의 판단 주체를 분리한다.** 실행 Coordinator가 skip/resume/new/publish를 결정하고, 저장 계층은 조회·검증·읽기·쓰기·게시를 수행한다. 논문별 Executor는 계산을 담당한다.
6. **F2의 게시된 실행 기록은 MLflow를 기준으로 한다.** 완료 신호는 네이티브 `FINISHED`다. 별도 완료 상태 DB, `durable_complete` 같은 중복 완료 플래그를 만들지 않는다. 단, 필요한 산출물의 존재·무결성은 읽는 시점에 검증할 수 있다.
7. **재현 데이터와 연구 근거는 보존한다.** 코드 호환성과 데이터 보존은 다르다. 기존 MLflow Run, 원본 코퍼스, 검증된 체크포인트는 삭제하지 않고, 재사용 여부를 명시적으로 판정한다.

### 성공 기준

- F2 논문 하나 추가 시 기존 W2V1/W2V2, DLFS 및 범용 패키지를 수정할 필요가 없다. **새로운 계산 기술 자체가 공통 패키지에 필요한 경우는 예외**다.
- F2의 완료 Run 조회·중복 실행 방지·로컬 복구·게시 판단에 각각 **하나의 책임 주체**만 있다.
- 학습 없이 저장된 모델로 평가와 논문별 분석을 수행한다.
- 같은 준비 데이터와 동일 계산 산출물을 논문 간 재사용할 수 있다.
- 결과 파일의 기본 경로가 안정적이며, 무조건 새 결과 파일을 누적하지 않는다.
- 코드 경로에 기존 포맷 호환 로직이 남지 않는다. 기존 영구 데이터는 보존하거나 명시적 변환 계획을 따른다.

---

## 1. 책임 계층

| 계층 | 소유권 | 담당 | 담당하지 않는 것 |
|---|---|---|---|
| `packages/repro-core` | 범용 라이브러리 | 설정 파싱, 선택·계획, 경로, 순수 로컬 I/O 유틸, 계산·저장 중립 계약 | F2의 MLflow 완료 판단, Word2Vec 학습 조건 |
| `packages/repro-mlflow` | 범용 MLflow 어댑터 | 인증된 클라이언트, 조회·게시·다운로드, 원자적 업로드 실패 처리에 필요한 저수준 기능 | 논문/테이블 분류, F2의 실행 재개 정책 |
| `packages/repro-io` | 범용 데이터 I/O | HTTP, S3, 해시, Common Crawl 프로토콜·아카이브 바이트 처리 | 코퍼스 선정·연구 필터·논문 정책 |
| `packages/w2v` | 독립 Rust 계산 라이브러리 | CBOW/SG, HS/NEG/NCE, NNLM, 학습 상태, 수치 계산 | MLflow·논문 테이블·F2 저장 경로 |
| `packages/deepscratch` | 독립 계산 라이브러리 | DLFS에서 쓰는 신경망 연산/Trainer | F2 리팩토링에 의한 강제 변경 |
| `studies/f2/execution` | **F2** | F2 실행 정의, 단일 Coordinator, 계획을 계산에 연결, 실행 결정·결과 처리 | 모델의 수치 연산, 저장소의 물리적 업로드 구현 |
| `studies/f2/storage` | **F2** | F2 Run 저장 배치, 체크포인트 검색, MLflow 게시·결과 선택 정책, 분석 캐시 연결 | 어떤 모델/실험 조건이 과학적으로 타당한지 판정 |
| `studies/f2/common` | F2 공통 | 연구 통계, 분석 산출물 규칙, F2 범위의 공통 자원/경로 조합 | Word2Vec에만 적용되는 연산 |
| `studies/f2/suites/common/w2v` | W2V 논문군 | W2V 준비·학습 연결·lookup·체크포인트 어댑터·평가 연산 | W2V1/2의 논문별 목표치·테이블 정책, 중복 MLflow lifecycle |
| `studies/f2/suites/w2v1`, `w2v2` | 개별 논문 | 실험 YAML, 고유 제약·설정, 논문 고유 단계, 표별 분석 | 공통 학습·저장·Run 검색을 재구현 |
| `studies/f2/corpus`, `studies/f2/catalog` | F2 데이터/연구 도메인 | 코퍼스 릴리스, 연구용 lineage·DDL, 원문 목표·정적 연구 카탈로그 | 실행 완료·체크포인트 상태를 별도 DB로 판정 |
| `studies/f2_cc` | 독립 Study | Common Crawl 생산·샘플링·감사·자체 DB | F2 논문 Executor 역할 |
| `studies/dlfs` | **레거시** | 현재 실행·저장·분석 방식 유지 | 신규 F2 실행 계약 이행 의무 |

**중요한 경계:** `repro-core`에 이미 있는 Planner 등의 순수 도구는 F2에서도 사용할 수 있으나, DLFS가 사용 중인 공통 Runner를 F2에 맞추기 위해 수정하지 않는다. F2의 새 실행 Coordinator는 F2 내부에 둔다. 범용 패키지의 수정은 실제 공통 기능이 필요하고 기존 소비자에게 미치는 영향이 정리된 경우에만 한다.

---

## 2. F2 목표 디렉터리 구조

아래 트리는 **책임 배치도**다. 파일명은 권장안이며 각 파일의 기능이 실제로 필요할 때만 만든다. 비어 있는 프로토콜/어댑터/테스트 파일을 일괄 생성하지 않는다.

```text
studies/f2/
├── catalog/                          # 정적 논문·자원·목표 데이터
├── src/f2/
│   ├── plugin.py                     # F2 진입점만
│   ├── cli/                          # plan / run / check / analyze / data 명령 조합
│   ├── execution/
│   │   ├── definitions.py            # F2 Suite 등록·실행 정의
│   │   ├── coordinator.py            # 실행 생명주기 유일한 정책 주체
│   │   ├── identity.py               # F2 실행 조건·attempt 식별
│   │   └── resources.py              # 입력 준비·검증의 조정
│   ├── storage/
│   │   ├── runs.py                   # 로컬 run 위치·상태·복구 대상 조회
│   │   ├── publication.py            # F2 -> MLflow 게시, 검색·검증
│   │   └── analysis_cache.py         # 분석용 MLflow 스냅샷/캐시 연결
│   ├── common/
│   │   ├── analysis/                 # 논문 중립 비교·표·렌더링
│   │   ├── stats/                    # F2에서 재사용하는 통계
│   │   └── paths.py                  # F2 path 조합 (범용 경로 객체는 core)
│   ├── corpus/                      # 기존 코퍼스 도메인 보존
│   ├── catalog/                     # 정적 catalog 파싱/검증
│   ├── preflight.py                 # F2 공통 서비스 사전 점검
│   └── suites/
│       ├── common/
│       │   └── w2v/
│       │       ├── spec.py           # W2V군 공통 설정·검증
│       │       ├── preparation.py    # vocabulary, phrase, 학습 입력 준비
│       │       ├── training.py       # Rust 엔진 연결, 학습 이벤트 조정
│       │       ├── checkpoints.py    # W2V/NNLM 상태 직렬화·복원 어댑터
│       │       ├── lookup.py         # lookup 산출물 계약
│       │       ├── evaluation.py     # word/phrase analogy 등 재사용 평가
│       │       ├── observations.py   # 공통 관측값 형식·수집
│       │       └── analysis.py       # 공통 요약 계산 (필요한 것만)
│       ├── w2v1/
│       │   ├── config/              # 원 논문·재현 조건 YAML
│       │   ├── spec.py              # W2V1 고유 검증과 자원 계약
│       │   ├── executor.py          # W2V1 고유 계산 순서/관측 정책
│       │   ├── resources.py         # W2V1 전용 자원만
│       │   └── analysis/
│       │       ├── table02.py
│       │       ├── table03.py
│       │       ├── ...
│       │       └── diagnostics/     # 테이블 분석과 별개 진단·코퍼스 조사
│       └── w2v2/
│           ├── config/
│           ├── spec.py
│           ├── executor.py          # word/phrase 등 고유 단계만
│           ├── preflight.py         # 6B/33B 고유 자원 추정만
│           ├── paper.py             # 논문 원본 표·질의·목표치
│           └── analysis/
│               ├── table01.py
│               ├── table02.py
│               ├── table03.py
│               ├── table04.py
│               ├── table05.py
│               ├── table06.py
│               └── section04_1.py
└── tests/                            # 실제 불변 조건과 핵심 경로만 검증
```

**파일 분할 기준**

- 표별 분석은 별개 파일을 기본값으로 한다. 여러 표가 같은 학습 Run이나 계산 결과를 사용하는 것은 허용한다.
- 공통 평가 연산을 표별 파일에 복제하지 않는다. 반대로 논문 목표치, 샘플 선정, 결과 해석은 공통 평가 모듈에 넣지 않는다.
- 길고 책임이 나뉜 파일만 패키지로 분할한다. 단순한 1:1 파일 대응을 위해 모듈을 생성하지 않는다.
- `suites/common/fasttext`, `suites/common/transformer` 등은 **실제 공유 구현이 생기기 전까지 만들지 않는다**.
- `f2/execution`과 `f2/storage`는 각각 *정책 경계*다. `repro-core.storage.LocalStore`나 MLflow API를 그대로 복제하는 장소가 아니다.

---

## 3. 실행 모델

### 3.1 실행 단위

- **Suite**: 논문/재현 대상 (`w2v1`, `w2v2` 등). 하나의 논문 내 여러 실험을 소유한다.
- **Experiment**: 논문에서 정의된 실험 묶음 (`e01`, `e02` 또는 테이블 단위 등).
- **Condition**: 하나의 원자적 실험 조건. 코퍼스, 모델, 목적 함수, 차원, 학습 스케줄 등 과학적으로 중요한 설정 포함.
- **Seeded Run**: Condition과 시드가 결합된 동일성 판단 단위.
- **Attempt**: 한 Seeded Run을 실제 계산/재개/재게시한 실행 시도. 명시적 rerun은 새 Attempt다.

**정체성은 경로·파일명에서 역추론하지 않는다.** `w2v1/spec.py`가 현재 파일명으로 table ID를 추정하거나 `w2v/tracked.py`가 executor 모듈 문자열에 `w2v2`가 포함됐는지 검사하는 방식은 제거 대상이다. 식별자는 정규화된 실험 명세에서 직접 만든다.

### 3.2 공통 실행 흐름

```text
CLI 요청
  -> F2 Suite 선택
  -> Planner: YAML + variant + seed를 정규화
  -> Condition Identity 확정
  -> Coordinator: 기존 MLflow FINISHED / 로컬 상태 조회
      |- 같은 조건의 게시 완료 Run: skip / 조회 결과 반환
      |- 로컬 완료 결과, 게시 누락: 재학습 없이 게시
      |- 동일 조건의 재개 가능한 checkpoint: 복원 후 계산
      |- 신규 조건 또는 명시적 rerun: 신규 Attempt 생성
  -> Preflight / Resource Preparation
  -> 논문별 Executor -> 알고리즘군 어댑터 -> 계산 엔진
  -> 로컬 result + 필수 산출물 검증
  -> Publication -> MLflow FINISHED
  -> 실행 결과 반환
```

- `plan`: 네트워크·훈련 없이 실행 목록과 필요 자원·비용 추정을 제공한다. 필요하다면 읽기 전용 메타데이터를 사용하되 실제 코퍼스 변환은 하지 않는다.
- `run`: 비용 승인, 실행 여부 결정, 입력 준비, 계산, 로컬 기록, 게시까지 담당한다.
- `check`: `plan`과 동일한 Condition Identity를 기준으로 MLflow 완료 상태를 대조한다. 별도 완료 집계 규칙을 만들지 않는다.
- `analyze`: 독립 실행. 학습/재개/게시를 호출하지 않는다.
- `evaluate`: 산출물 + 명시적 평가 자원으로 평가 결과 생성. 필요하면 분석 파이프라인의 공통 평가 단계로 사용한다.

### 3.3 컴포넌트별 책임

| 컴포넌트 | 입력 | 출력 | 금지 |
|---|---|---|---|
| Planner | YAML, 선택 조건, 시드 | 검증된 계획 및 Condition 정보 | MLflow 조회, 코퍼스 대량 materialize, checkpoint 복원 |
| Coordinator | 계획, 실행 옵션, 저장 조회 결과 | skip/resume/run/publish 결정과 최종 receipt | 수치 계산, 논문별 table 분기, 원시 MLflow API 상세 구현 |
| Resource Resolver | 명세에 선언된 자원 요구 | 검증된 입력 참조/준비 결과 | 학습 수행, 임의 코퍼스 교체, 별도 완료 상태 관리 |
| Suite Executor | 정규화된 spec, 준비된 자원, 실행 Context | 수치 계산 산출물, 메트릭, checkpoint 상태 | MLflow Run 검색·생성·종료, 로컬 Attempt 정책 결정 |
| Engine | 학습 데이터·수치 설정 | 계산 결과·학습 상태 | F2 경로/CLI/논문 목표 해석 |
| Publication | 검증된 Run 산출물·메타데이터 | MLflow Run ID 또는 명시적 실패 | 훈련 재실행, 산출물 재계산 |
| Analyzer | 선택된 게시 Run, 평가 자원, 분석 설정 | 논문별 표·그림·보고서 | 학습, checkpoint 복원 후 추가 학습, 입력 복구 시도 |

**중복 실행 판단 금지:** `run`, `check`, `download_results`는 동일한 Condition Identity 및 Run 선택 규칙을 이용한다. 조회 API는 하나로 합치고 호출 목적만 구분한다. 조회할 때 최신 Run을 자동으로 섞어 서로 다른 학습 설정/시드를 하나의 조건으로 취급하지 않는다.

### 3.4 공통 계약의 최소 필드

| 계약 | 공통 부분 | 논문별 부분 |
|---|---|---|
| Suite Definition | Suite ID, 설정 디렉터리, 실행 엔트리포인트, 분석 엔트리포인트 | 실제 YAML과 논문별 기능 |
| Run Spec | Suite/experiment/condition ID, seed, 입력 참조, semantic config digest, backend/algorithm revision | 모델·목적 함수·전처리·스케줄 |
| Execution Context | 준비된 입력 참조, 작업 디렉터리, 관측값 기록·체크포인트 작성 인터페이스, 진행 리포터 | 계산 엔진에 전달할 설정 |
| Execution Result | 필수 산출물 목록, metric 기록, 결과 메타데이터, 실행 종료 여부 | tensor/embedding/모델 상태의 내용 |
| Checkpoint Codec | 상태 저장, 상태 로드, 동일성·완전성 검증 | W2V/NNLM/미래 모델의 실제 직렬화 |
| Analysis Request | Suite, table, Run 선택, benchmark 선택, 분석 옵션, overwrite 여부 | 테이블별 분석·출력 형식 |

이를 위해 새 프레임워크를 만들 필요는 없다. 현행 `ExecutionDefinition`, `Planner`, `LocalStore`, `CheckpointManager`, `MlflowArtifactCache` 등의 기존 기능을 **필요한 책임만 남겨 재사용**한다. F2의 기존 계약과 중복되는 래퍼는 마이그레이션 완료 뒤 삭제한다.

### 3.5 계산과 전처리

- 준비 데이터는 실행 전에 확정하지만, 이미 게시된 동일 조건을 skip할 경우 불필요한 전처리를 먼저 수행하지 않는다.
- 데이터 자원은 `source_id`, revision, content identity, 변환 규칙을 통해 선택한다. 준비 파일의 임의 로컬 절대 경로는 과학적 동일성의 유일한 근거가 아니다.
- W2V2의 word/phrase는 같은 공유 학습 엔진을 사용한다. phrase 생성·문맥 정책·대규모 자원 추정처럼 다른 단계만 W2V2에 남긴다.
- CPU/GPU 검증은 해당 Suite 요구 조건을 사용한다. F2의 CPU parity 제한을 DLFS 또는 모든 미래 Suite의 전역 제한으로 만들지 않는다.
- 6B/33B 등 비싼 실험은 명시적 승인 없이는 학습하지 않는다. dry-run/plan은 훈련을 수행하지 않는다.

### 3.6 신규 논문 확장 계약

새 논문을 추가하는 기본 절차는 **(1) 자원·평가 요구 정의 → (2) YAML 명세 → (3) 필요한 계산 어댑터 연결/추가 → (4) 논문별 분석 → (5) Suite 등록**이다. 기존 논문의 `executor.py`, `analysis.py`, YAML을 수정해서 새 논문을 연결하지 않는다.

서로 다른 논문이 공유하는 학습·평가 방식이 생기면 논문군 공통 영역으로 옮긴다. 하나의 논문에서만 사용하는 기능을 추상 인터페이스로 포장하지 않는다.

---

## 4. 저장 모델

### 4.1 저장 자원과 권위

| 저장 영역 | 주 데이터 | 권위/수명 | 중복 방지 기준 |
|---|---|---|---|
| `workspace/inputs/` | 로컬 코퍼스, 벤치마크 | 검증된 입력·준비 데이터의 재사용 공간 | 원본 revision + 변환 규칙 + 검증된 content identity |
| F2 PostgreSQL | 연구용 코퍼스 lineage·메타데이터 | 코퍼스 도메인 기록 | 기존 스키마의 자원 ID·제약 |
| 외부 S3 | 코퍼스 바이트·샤드 | 외부 객체의 영구 저장 | URI + digest + immutable version |
| `workspace/runs/` | 진행 중 Run, checkpoint, 완료 직후 결과 | 복구 가능한 로컬 실행 상태 | Condition + Attempt |
| MLflow | 게시된 실행 결과·메트릭·필수 모델 산출물 | **F2의 영구 실험 실행 기록** | Condition 조회 + 개별 MLflow Run ID |
| `workspace/analysis/cache/mlflow/` | MLflow에서 내려받은 산출물 | 재생성 가능 | tracking server key + MLflow Run ID + artifact identity |
| `workspace/analysis/cache/results/` | 계산량 큰 평가·통계 결과 | 재생성 가능 | 입력 Run ID/benchmark digest/평가 옵션/코드 revision |
| `workspace/analysis/results/` | 최종 표·그림·문서 | 명시적 갱신 대상 | 논문·테이블·필터별 **고정 경로** |

**하나의 보편적인 ‘저장 DB’는 만들지 않는다.** 연구 코퍼스 메타데이터, Run 게시 결과, 로컬 복구 상태는 서로 다른 목적을 가진다.

### 4.2 목표 Workspace 구조

```text
workspace/
├── inputs/
│   ├── f2/
│   │   ├── corpus/                     # 원본/참조 파일
│   │   ├── prepared/                   # 재사용 가능한 검증된 변환 결과
│   │   └── benchmarks/                 # 평가 자원
│   └── dlfs/                           # 기존 정책은 레거시 영역에서 해석
├── runs/
│   └── f2/
│       ├── w2v1/
│       │   └── <slot_id>/
│       │       └── <attempt_id>/
│       │           ├── config.json
│       │           ├── result.json
│       │           ├── checkpoints/
│       │           ├── metrics/
│       │           └── artifacts/
│       └── w2v2/
│           └── <slot_id>/<attempt_id>/...
└── analysis/
    ├── cache/
    │   ├── mlflow/<server_key>/<run_id>/...
    │   └── results/f2/<suite>/...
    └── results/
        └── f2/
            ├── w2v1/<table>/<corpus>/...
            └── w2v2/<table>/<corpus>/...
```

`<slot_id>`는 사용자에게 읽히는 실험 위치이고 `<attempt_id>`는 그 아래의 실제 실행 하나다. 이 경로는 일례이므로 기존 실험 식별자와 매핑한 뒤 확정한다. **MLflow Run ID는 게시 후 식별자**이므로 실행 시작 전 로컬 저장 경로를 MLflow ID에 종속시키지 않는다.

경로 별칭(`cache_root`, `staging_root`, `artifacts_root` 등)은 신규 F2에서는 쓰지 않고 `inputs_root`, `runs_root`, `analysis_cache_root`, `analysis_results_root`를 직접 사용한다. DLFS를 위해 존재하는 레거시 API는 이번 작업이 사용·확장하지 않는다.

### 4.3 Condition Identity와 Attempt Identity

**Condition Identity**의 의미상 입력:

- Suite·실험 ID·원자 조건 ID·seed.
- 학습 결과에 영향을 주는 정규화된 설정(모델/목적 함수/스케줄·분포 처리·전처리).
- 검증된 입력 리비전 및 처리 정책의 identity.
- 결과 동일성에 영향을 미치는 엔진·알고리즘 구현 revision, precision 및 필요한 런타임 설정.

**포함하지 않을 값:** MLflow 서버 URL, 로컬 경로 문자열 자체, 표시용 run name, 진행 상황 출력 빈도, 분석 출력 디렉터리. 하드웨어·스레드 정책이 실제 수치 결과에 영향을 줄 수 있으면 예외 없이 의미 설정/재현성 메타데이터로 명시한다.

**Attempt Identity**는 같은 Condition의 물리적 실행을 구별한다. resume은 같은 Attempt를 이어가고, `--override`의 재실행은 새 Attempt로 기록한다. MLflow Run ID는 게시된 Attempt와 1:1 대응한다.

`planned_run_slot_id`만으로 동일성을 판단해서는 안 된다. 동일 slot에 설정 변경이 있을 수 있으므로 **slot + semantic digest + seed + 필요한 입력/알고리즘 revision**을 비교한다. 비교 규칙은 한 곳에서 유지한다.

### 4.4 실행·게시 상태 전이

```text
[no matching published result]
  -> [no local attempt] -----create----> RUNNING
  -> [valid local checkpoint] -resume--> RUNNING
  -> [local complete result] ----------> READY_TO_PUBLISH

RUNNING -> LOCAL_COMPLETE -> READY_TO_PUBLISH
READY_TO_PUBLISH -> PUBLISHING -> MLflow FINISHED
                               -> 실패: LOCAL_COMPLETE 보존 / MLflow FAILED

[matching MLflow FINISHED] -> SKIPPED / reference returned
[explicit --override]     -> new Attempt (old record preserved)
```

- `RUNNING`, `LOCAL_COMPLETE` 등은 **개념적 상태**다. 새 상태 DB나 중복 완료 태그를 만들라는 뜻이 아니다. 로컬 파일 존재·검증 결과, MLflow 상태에서 도출한다.
- 로컬 checkpoint는 과학적으로 필요한 상태(모델·optimizer·RNG·진행 위치)를 저장한다. 직렬화 내용은 알고리즘군 codec이 소유하고, 저장 위치와 retention은 저장 계층이 소유한다.
- 완료 로컬 산출물의 게시가 실패한 경우, 재학습 없이 다시 게시 가능해야 한다.
- MLflow `FINISHED` 전에 필수 산출물 업로드를 끝내야 한다. `FINISHED`인데 필요한 artifact가 없으면 Analyzer가 데이터를 임의로 보충하지 않고 오류/불완전 상태를 보고한다.
- 로컬 실행 데이터는 자동으로 제거하지 않는다. 사용자가 명시적으로 정리하는 정책을 별도로 두며, 재개 가능한 상태나 게시 실패 데이터를 정리 대상에 섞지 않는다.

### 4.5 Artifact 계약

**공통적으로 필요한 메타데이터**는 실행 조건, 실제 seed, 모델/입력 revision, artifact 경로·크기·필요한 무결성 정보, 관측값 경로, 완료 여부다. 실제 바이너리 포맷은 알고리즘군이 결정한다.

- W2V/NNLM: 입력 임베딩 lookup과 vocabulary 순서, 모델 상태, epoch 관측값, phrase lineage(해당할 때만).
- 향후 다른 모델: 각자의 모델 체크포인트나 평가용 출력. 불필요한 임베딩/lookup 형식을 강제하지 않는다.
- 업로드 대상은 실행 산출물의 **명시된 목록**으로 결정한다. 디렉터리의 모든 파일을 무분별하게 게시하지 않는다.
- 큰 checkpoint를 lookup 산출물 대신 재활용하는 것은 가능하나, **Analyzer가 접근할 수 있는 결과 인터페이스**는 유지해야 한다. 내부 물리 저장 방식이 평가 규약을 바꾸면 안 된다.
- 동일 검증 정보를 수행 과정마다 별도 파일로 증식시키지 않는다. 이미 있는 manifest를 활용하며, 독립된 완료 권한을 가진 별도 레지스트리를 만들지 않는다.

### 4.6 Prepared data 캐싱

- 준비 데이터의 캐시 키는 원본 데이터 identity + 정규화/phrase/vocabulary 등의 변환 정책 + 필요 알고리즘 revision으로 만든다.
- 두 논문이 같은 자원과 같은 변환 정책을 사용하면 **같은 prepared artifact**를 참조한다.
- 준비 중 장애가 발생했을 때 불완전 파일이 완성본처럼 보이지 않도록, 임시 생성 후 검증된 결과를 게시한다.
- 캐시 재사용을 위해 매 Run마다 6B/33B 원본 전체를 다시 해시해야 하는 구조는 피한다. 검증된 불변 manifest와 content identity를 신뢰하고 필요한 경계에서 검증한다.
- 존재만 확인하고 검증하지 않은 파일을 재사용하지 않는다.

### 4.7 분석 실행과 결과 경로

1. 요청한 논문/테이블/코퍼스/시드 집합과 정확한 Condition을 확정한다.
2. MLflow에서 Run을 조회하거나 `--offline`이면 기존 다운로드 캐시만 사용한다.
3. 게시된 모델 산출물, benchmark 파일, 평가 정책을 확인한다.
4. 동일 분석 키의 계산 캐시를 재사용하거나 계산한다.
5. 논문별 Analyzer가 최종 표/그림을 작성한다.

**하나의 테이블·필터 조합은 하나의 안정적인 보고서 위치**를 갖는다. 예: `analysis/results/f2/w2v1/table03/wmt/summary.md`. 실제 결과 묶음은 해당 테이블이 필요로 하는 JSON/CSV/PNG만 포함한다. 불필요한 실행별·분석별 무한 파일 복제는 하지 않는다.

- 결과 파일이 존재해도 입력 fingerprint가 달라졌으면 그대로 재사용하지 않는다.
- 분석 조건이 달라졌을 때는 명시적 `--override` 또는 `--overwrite`로 재생성을 승인받는다. **동일 경로의 출력 갱신**이며 자동 버전 파일 생성이 아니다.
- 불완전 입력은 `unavailable`로 보고하고, 빈 데이터를 정확도 0으로 취급하지 않는다.
- 여러 논문·테이블에서 같은 Run을 재사용할 수 있다. 분석이 Run의 identity를 바꾸지 않는다.
- 논문 목표치(paper reference)와 재현 측정치(observation)는 자료구조/보고서에서도 분리한다.

---

## 5. 현재 코드 → 목표 코드: 책임별 마이그레이션 지도

아래 판정은 `dev`의 확인된 파일 위치와 주요 함수에 근거한다. **최종 파일명은 확정 계약이 아니라 후보**이며, 대규모 파일은 실제 내부 책임별로 분할한다. 이동·삭제 전에 모든 실제 import/call site와 테스트 참조를 확인한다.

| 현재 | 확인된 책임/문제 | 목표 조치 |
|---|---|---|
| `f2/definition.py`, `f2/suites/protocol.py` | F2 등록과 `SuiteCatalog`가 실행 정의를 별도로 조합 | `f2/execution/definitions.py`에서 등록·정의 책임 통합. 필요한 기능을 옮기고 원래 모듈 삭제 |
| `f2/cli.py` | run/check/analyze/평가·phrase preflight가 한 파일에 집중, `check`가 별도 완료 조회 | `f2/cli/`로 명령별 분리. 비즈니스 로직은 `execution`/분석으로 내려 CLI는 얇게 유지 |
| `f2/suites/w2v/tracked.py` | MLflow 완료 중복 조회, corpus 준비, Run 실행 조합 | 공통 `f2/execution/coordinator.py` 및 resource 조정으로 해체. W2V 추정 분기 삭제 |
| `f2/suites/w2v/executor.py` | 학습뿐 아니라 skip/resume/new, local path, MLflow 게시까지 수행 | lifecycle을 Coordinator/Storage로 이동. W2V 학습 연결은 `suites/common/w2v/training.py`로 이동 |
| `f2/suites/w2v/tracking.py`, `f2/tracking.py` | MLflow 검색·게시와 tracking URI 해석이 분산 | 저수준 MLflow 기능은 `repro-mlflow`; F2 조회·선택·게시 규칙은 `f2/storage/publication.py`에 모음. URI 해석은 F2 단일 위치로 정리 |
| `f2/run_identity.py`, `f2/suites/w2v/spec.py` | slot/tag와 config digest 생성이 분산 | Condition/Attempt 계약을 `f2/execution/identity.py`에서 일관 정의. W2V 고유 semantic fields는 논문군 spec에 유지 |
| `f2/suites/w2v1/spec.py` | table ID 및 spec ID를 파일명과 조건에서 추정 | 명시적 메타데이터만으로 식별. 논문 고유 설정 검증은 유지 |
| `f2/suites/w2v2/spec.py` | word/phrase/6B/33B 연구 정책을 검증 | 논문 전용으로 유지. 공통 lifecycle/저장과 혼합하지 않음 |
| `f2/suites/w2v/artifacts.py` | vocabulary 캐시, checkpoint codec, lookup 형식 등 여러 책임 집약 | `suites/common/w2v/{preparation,checkpoints,lookup}.py`로 책임별 분리. 내용·수치 의미 유지 |
| `f2/suites/w2v/{engine,observations,corpus,phrases}.py` | 엔진 연결·관측·입력 전처리 | `suites/common/w2v/`로 이동/분리, F2 논문 고유 phrase 정책은 W2V2에 남김 |
| `f2/suites/w2v/{evaluation,evaluate,analysis,readiness}.py` | 공통 평가와 공통 요약·완료도 평가 | 일반적인 기능은 `suites/common/w2v/` 및 `f2/common/analysis/`에 배치. 기능별로 실제 공유 범위를 확인 |
| `f2/suites/w2v/cache.py` | MLflow download, run 선택, W2V2 table 매핑, 로컬 캐시 독자 reader | 다운로드/선택/캐시를 `f2/storage/analysis_cache.py`에 분리하고 table 매핑은 각 Suite로 이동 |
| `f2/suites/w2v1/{analysis,table3,...,table7}.py` | Table 2만 analysis.py, 나머지 표는 루트 파일 | `w2v1/analysis/table02.py`–`table07.py`로 정리. 공통 metric 계산은 재사용 |
| `f2/suites/w2v1/{corpus_comparison,corpus_evidence,corpus_investigation,corpus_report,corpus_statistics,diagnostic,relations}.py` | 표별 분석과 별도 코퍼스 조사·진단 혼재 | `analysis/diagnostics/` 또는 `analysis/common/`으로 역할별 정리; 표별 분석과 재사용 관계 확인 후 결정 |
| `f2/suites/w2v2/analysis.py` | 1,000여 줄에 여러 표 분석·다운로드 가정·출력 수행 | 테이블별 Analyzer로 분할, 공통 조회/출력/평가를 중복하지 않음 |
| `f2/suites/w2v2/{paper,preflight,executor}.py` | 논문 목표, 대규모 자원 정책, phrase 학습 특수화 | 논문 고유 부분은 유지. Executor는 계산 정책만 남김 |
| `f2/common/{analysis,stats,paths}.py` | F2 전체에 재사용되는 도구 | 실제 범위에 맞게 유지. W2V 전용 구현이 섞이면 논문군 공통으로 이동 |
| `f2/{corpus,catalog}` | 코퍼스/연구 카탈로그, 일부 runtime identity 중복 | 도메인 책임 유지. 정적 catalog와 YAML이 중복 정의하는 실행 값은 단일 출처로 정리 |
| `repro-core/execution/{planning,runner,locator}.py` | 범용 Planner와 DLFS가 사용하는 Runner/Locator가 결합 | Planner 등 순수 메커니즘은 재사용. **기존 Runner의 cleanup 로직을 F2 lifecycle에 사용하지 않음**. DLFS 동작을 위해 무리하게 변경하지 않음 |
| `repro-core/context/paths.py` | `workspace`와 레거시 `staging_root/cache_root/artifacts_root` alias 공존 | F2 신규 코드에서는 명시적 새 경로 API만 사용. 레거시 별칭을 F2에서 추가·복제하지 않음. DLFS의 기존 API 일괄 제거는 이번 비범위 |
| `repro-core/storage/local_store.py`, `context/checkpoint.py` | 로컬 쓰기/체크포인트 보존 수단 | 일반적인 메커니즘으로 재사용. F2 정책은 별도 F2 계층에서 결정 |
| `repro-mlflow/artifact_cache.py`, `results.py`, `runtime/` | MLflow 저수준 사용과 DLFS 스키마 경로가 공존 | 기존 재사용 가능한 client/cache 기능 활용. F2 구현을 위한 DLFS 스키마 강제·호환층은 만들지 않음 |
| `packages/w2v/src/` | Rust W2V·NNLM 계산 구현 | 원칙적으로 그대로 재사용. 실험 연구 정책과 무관한 버그·공통 인터페이스 수정만 분리 수행 |
| `studies/dlfs/` | 자체 실험·CLI·Executor·저장 관행 | **레거시로 유지**, F2 계약 이식하지 않음 |
| `studies/f2_cc/` | 별도 Common Crawl producer | 유지. 검증된 release를 F2에 전달하는 경계만 명확히 함 |

### 특히 확인된 현재 설계 결함

1. `w2v/tracked.py`와 `Word2VecExecutor.execute()` 둘 다 MLflow 완료 Run을 찾는다. `f2/cli.py::check()`도 독자적 완료 판정(현재는 slot ID 위주)을 한다.
2. 공통 `repro-core/execution/runner.py`는 `durable_complete` 속성을 기준으로 로컬 staging을 제거하는 코드가 있다. F2의 보존·복구 정책과 섞으면 안 된다.
3. `w2v/cache.py::download_results()`는 최근 학습 설정을 기준으로 run을 선택하고, W2V2 테이블 매핑도 포함한다. 입력 identity가 고정된 분석 요청으로 분리해야 한다.
4. `w2v1/spec.py`는 파일명으로 table을 판단하고, `w2v/tracked.py`는 executor module 문자열로 Suite를 추정한다.
5. `w2v2/analysis.py`가 여러 표의 평가·집계·보고를 한꺼번에 수행한다. `w2v1`의 table 분석 위치도 불규칙하다.
6. W2V2 전처리 캐시·대규모 preflight는 고유 실험 정책이므로 모든 Suite의 일반 조건으로 승격하면 안 된다.

---

## 6. F2의 실험 Catalog와 YAML 관계

현재 `studies/f2/catalog/w2v.json`에는 paper targets, resources, resource versions, execution plans, planned slots 등이 있고, 각 Suite YAML에도 실행 설정과 identity 정보가 있다. 이중 정의를 계속 확대하지 않는다.

**권장 소유권:**

- **Catalog:** 원 논문의 대상/표/수치, 자원 출처·검증 리비전, 원 논문과 로컬 재구성 계획의 관계.
- **Suite YAML:** 실제 실행에 필요한 **정규화 가능한 수치 설정**, variant·seed 정책·연산/전처리 정책.
- **Planner:** 두 자료를 확인해 하나의 immutable 실행 요청을 만든다. 필요한 식별자는 *여기서 한 번만* 확정한다.
- **MLflow:** 실제로 실행된 설정과 산출물을 기록한다. Catalog는 MLflow의 완료 Run 테이블을 복제하지 않는다.

동일 파라미터가 두 파일에 존재할 때 어느 쪽을 우선하는지 암묵적으로 두지 않는다. 코드의 중복 선언을 정리하면서 단일 출처를 지정하고, 필요한 검증 관계만 남긴다.

---

## 7. 마이그레이션 절차 (구현 순서)

| 단계 | 작업 | 완료 조건 |
|---|---|---|
| **M0. 현행 계약 고정** | F2의 CLI, 정규화된 실행 조건, 완료 Run 매칭, 3가지 복구 경로, MLflow 산출물 형식, 분석 결과를 현행 코드와 소규모 fixture로 기록 | 무엇을 보존할지 기준이 확정됨. 대규모 학습·실서비스 기록 변경 없음 |
| **M1. 실행 identity 단일화** | `run_identity`, W2V spec digest, planned slot, `check` 조회를 하나의 조건 계약으로 정리. 파일명/모듈명 추론 제거 | plan/run/check에서 같은 조건이 같은 identity를 가짐 |
| **M2. F2 Coordinator 분리** | F2 `tracked.py`/W2V Executor의 skip/resume/publish 판단을 Coordinator에 모으고 기존 경로 삭제 | 동일 결정을 여러 모듈에서 하지 않음. DLFS Runner 영향 없음 |
| **M3. 저장 분리** | local attempt, checkpoint 발견, MLflow 검색/게시, 결과 다운로드를 F2 storage 경계로 이동. W2V codec 유지 | 로컬 완료 → 게시/실패 재시도, 중단 → 복구, 게시 Run → skip 경로 일치 |
| **M4. W2V 논문군 공통화** | `suites/w2v/`를 `suites/common/w2v/`로 재편하고 계산·전처리·checkpoint·lookup·평가를 책임별 정리 | W2V1/2가 동일 공통 구현을 사용하고 기존 계산 결과 의미 유지 |
| **M5. 논문별 분석 정리** | W2V1/2 table 파일 분리, 공통 평가·다운로드·집계는 재사용, 결과 경로 통일 | 학습 없이 한 테이블만 분석 가능, 동일 요청 재활용/명시적 갱신 |
| **M6. CLI·문서 정합화** | F2 CLI를 얇게 나누고 오래된 아키텍처 문서 교정, 이전 코드·모듈·테스트 참조 제거 | 신규 문서가 현재 코드와 맞고 삭제된 API를 참조하지 않음 |
| **M7. 최종 검증·전환** | 작은 결정적 계산, 저장·복구·게시, 두 논문 table 분석을 검증. 예전 경로 제거 | 새 경로만 남고 불필요한 호환 분기 0개, 실제 결과 보존 |

**진행 원칙:** 각 단계는 기존 구현과 새 구현을 병렬로 영구 유지하지 않는다. 단계 내에서 호출부를 함께 이전하고 기존 경로를 제거한다. 코드 변경과 데이터 삭제는 별개 작업이다. 기존 영구 데이터에 대한 대량 일회성 변환은 수행 목록과 검증 기준을 사전에 명시한 뒤 별도 승인받는다.

### 마이그레이션과 데이터 처리

- **기존 MLflow Run:** 읽을 수 있는 형식이면 현재 연구 이력으로 보존한다. 신규 코드에 과거 스키마 읽기 분기를 만들지 않는다. 새 분석 입력으로 사용할 필요가 있으면 일회성 변환 산출물을 마련하거나 해당 Run을 과거 기록으로만 유지한다.
- **기존 로컬 checkpoint:** 호환되지 않으면 자동 resume하지 않는다. 동일 형식/정체성을 증명할 수 있을 때만 새 구조로 변환한다.
- **준비 코퍼스:** 바이트가 동일하고 lineage가 검증되면 이동 없이 참조할 수 있다. 재학습을 강제하지 않는다.
- **분석 캐시·보고서:** 재생성 가능하면 최신 경로로 재계산한다. 영구 과거 결과를 조용히 삭제하지 않는다.
- **DLFS:** 별도 마이그레이션 대상으로 삼지 않는다. 기존 진입점과 저장 규약을 그대로 둔다.

---

## 8. 문서 관리 개편

| 문서 | 조치 | 이유 |
|---|---|---|
| `AGENTS.md` | F2의 새 단일 실행·저장 규약, DLFS 레거시 범위를 명시하고 중복/충돌 규칙 제거 | 에이전트가 무엇을 준수할지 확실해야 함 |
| `docs/architecture/overview.md` | Workspace 경로, F2/DLFS 분리, 현대 패키지 역할로 갱신 | 옛 `.staging/.cache/data/artifacts` 설명과 충돌 |
| `docs/architecture/studies.md` | F2 실제 W2V1/2 및 향후 Suite 구조, 독립 F2-CC, DLFS 레거시 설명 | 오래된 예상 Suite 구조 교체 |
| `docs/architecture/packages.md`, `adapters.md` | 실제 API/모듈·테스트 경로에 맞춰 정정 | 없는 파일·과거 패키지 구조가 남아 있음 |
| `docs/guidelines/path-and-storage.md` | 단일 `workspace` 경로 설명과 F2 게시 상태 계약으로 정리. 레거시 DLFS 규칙은 분명히 구분 | 한 문서에 신·구 저장 규약 혼재 |
| `docs/architecture/f2-experiments.md` | **새 규약을 설명하는 정식 F2 runtime 문서**로 재구성 | 변경 이력과 현재 필수 계약 혼합 |
| `docs/guidelines/execution-and-verification.md` | 현재 CLI/justfile 기준 명령과 테스트 정책 정정 | 오래된 명령 예제 제거 |
| `README.md`, `studies/f2/README.md` | 현재 진입점·설치·구조만 간결하게 유지 | 새 참여자에게 옛 구조를 설명하지 않도록 |
| `.agents/PLAN.md`, `ownership-inventory.md`, NNLM 최적화/재현 기록 | **과거 시점의 설계·실험 기록으로 분류**하고 현재 규칙 링크에서 분리 | 기록 보존과 현재 규칙 권위를 분리 |
| `studies/f2/catalog/*.md` | 연구 근거 보존, 현재 상태·TODO만 최신화 또는 이력으로 표시 | 과거 진행 상황과 현재 구현 혼동 방지 |
| `studies/dlfs/.../*.md` | 이번 설계로 구조를 고치지 않음. 실행 불가 명령 등 명백한 사실 오류만 별도 레거시 정비 | 범위 확대 방지 |

새 문서는 `docs/architecture/f2-runtime.md`처럼 **단일한 정식 설계 문서**로 둘 수 있다. 기존 계획 기록은 링크만 남기거나 archive로 분리한다. 문서를 추가하면서 동일 규칙을 여러 문서에 복사하지 않는다.

---

## 9. 검증 원칙

테스트의 목적은 구조를 유지하기 위해 테스트 수를 늘리는 것이 아니라, 잘못된 재사용·손실·중복 실행을 막는 것이다.

**최소 핵심 검증 시나리오**

1. 같은 YAML·seed·자원 identity → 같은 Condition; 과학적으로 의미 있는 설정 변경 → 다른 Condition.
2. 게시 완료 Run 존재 → 학습·대규모 전처리 없이 skip.
3. 로컬 완료 결과 존재, 게시 실패 → 재학습 없이 게시 재시도.
4. 재개 가능한 checkpoint → 실제 수치 상태·RNG·진행 위치 복원(지원하는 엔진에서), 호환되지 않으면 거절.
5. 명시적 rerun → 새 Attempt/MLflow Run, 기존 결과 보존.
6. 서로 다른 시드·다른 corpus·다른 semantic digest를 동일 조건 분석 결과로 섞지 않음.
7. MLflow artifact cache 삭제 뒤 다시 다운로드해 같은 평가를 수행할 수 있음.
8. 분석 offline 모드에서는 네트워크·학습 수행 없음; 누락 입력은 unavailable로 표현.
9. W2V1/2 대표적 작은 조건에서 산출물 값·의미를 기존과 비교, 원 논문 재현 데이터 해석 보존.
10. DLFS의 현행 진입점·실행은 F2 리팩토링에 의해 영향을 받지 않음.

테스트 파일을 모듈마다 1:1로 만들거나, 미구현 기능을 미리 검증하는 placeholder 테스트를 생성하지 않는다. 기존 중복·무의미 테스트는 통합/삭제한다. 루트 `just check`는 유용한 게이트지만, 500개 이상의 테스트를 무조건 유지·증가시키는 것을 성공 기준으로 삼지 않는다.

---

## 10. 구현 금지 항목

- F2와 DLFS를 억지로 하나의 Executor 인터페이스로 통합.
- 과거 API를 위해 영구적인 alias/shim/fallback 추가.
- `suites/common/` 아래 실제 사용처 없는 미래 알고리즘군 구조 선제 생성.
- F2 논문별 Executor에 MLflow Run 조회·종료·게시 구현을 복제.
- analyzer가 MLflow를 직접 고치거나 missing input을 임의로 재생성.
- 기존 Condition과 다른 새 결과를 동일 identity라고 조용히 재사용.
- 결과가 같을 때마다 새 분석 파일 생성, 예전 파일 무조건 중복 저장.
- 체크포인트나 영구 MLflow Run의 일괄 삭제/자동 마이그레이션.
- 문서 정리를 구실로 실제 실험 설정·원 논문 수치를 근거 없이 변경.
- 새 기능 구현 없이 추상 클래스·프로토콜·모듈 개수만 증가.

---

## 11. 미확정 사항과 기본 처리

이 문서는 구현 방향을 확정할 수 있는 수준이지만, 실제 코드 이전을 시작하기 전에 다음 사항만 **현행 사용처·저장 데이터를 통해 증거 기반으로 결정**한다. 새로운 사용자 결정을 요구하는 것은 아니다.

1. **기존 MLflow Run을 재분석할 필요가 있는지**: 필요한 Run 집합과 실제 형식만 점검하고, 필요 시 *일회성* 변환 설계를 별도 제시.
2. **기존 `ExecutionDefinition`/Planner의 재사용 범위**: 유지 가능한 공통 함수만 사용. DLFS Runner를 건드리지 않음.
3. **분석 최종 경로의 정확한 표기**: 테이블·코퍼스·분석 필터를 안정적으로 표현하도록 선택. 기존 보고서는 결과 파일로 보존하거나 재생성 가능성을 판단.
4. **W2V1 진단 모듈의 세부 소유권**: 현재 대형 분석 파일의 실제 호출 관계를 보고 table 분석/공통 분석/독립 연구 조사로 분류.
5. **체크포인트 직렬화 범위**: 기존 모델별 상태 형식과 resume 가능성을 확인한 뒤 공통 저장 메커니즘과 codec의 경계를 확정.

### 최종 규정

> **F2의 실행 생명주기는 하나, 저장 책임은 데이터 성격별로 분리, 수치 계산과 논문별 분석은 독립적으로 유지한다. DLFS는 레거시이며 F2 설계에 맞출 의무가 없다. 호환 코드를 남기지 않고 필요한 것은 한 번 마이그레이션한다.**

이 문서는 새 구현이 도달해야 할 **목표 구조**를 정의한다. 기존 코드를 보존해야 한다는 암묵적 전제나, 제안된 파일 이름에 맞추기 위한 기계적 리팩토링을 허용하지 않는다.
