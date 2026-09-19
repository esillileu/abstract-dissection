# w2v: word2vec C oracle and Rust port

이 패키지는 word2vec 학습 알고리즘을 분리한 모듈식 C 구현(`reference/`)과
그 동작을 옮긴 Rust 라이브러리(`src/`)를 제공합니다. 비교 기준은 **모듈식 C
구현**입니다. `reference/z_original_w2v.c`는 읽기 전용 원본 스냅샷이며
빌드·테스트 대상이 아닙니다. Rust는 작은 고정 입력의 단일 스레드 학습에서
모듈식 C와 동일한 임베딩 비트 해시를 냅니다. 이것이 원본 프로그램의 모든
입력·옵션·출력 형식과 호환된다는 뜻은 아닙니다.

## 현재 제공 범위와 서빙 가능 여부

| 기능 | Rust 포트 | 모듈식 C | 원본 스냅샷 |
|---|---|---|---|
| 어휘 구축, CBOW/Skip-gram 학습, 단일·다중 스레드 실행 | 제공 | 제공 | 제공 |
| HS 또는 negative sampling 선택 | 한 학습 실행에서 하나 선택 | 한 학습 실행에서 하나 선택 | 둘 다 동시에 켤 수 있음 |
| 빈도 높은 단어 서브샘플링 | 설정으로 적용 가능 | 설정으로 적용 가능 | `-sample`로 적용 가능 |
| 학습된 입력 임베딩 조회 | `Model::snapshot_into`와 `Vocabulary::find` | `model_snapshot`와 `vocab_find` | 학습 뒤 파일로 출력 |
| 임베딩·어휘 저장/로드, 실행 CLI, HTTP/gRPC 서비스 | 없음 | 없음 | 벡터 출력·어휘 파일 옵션을 갖춘 CLI; 서비스 API는 없음 |
| 바이너리/텍스트 벡터 파일, K-means 단어 클래스 | 없음 | 없음 | 출력 옵션으로 제공 |

**현재 상태:** Rust 라이브러리는 학습 완료 후 같은 프로세스에서 입력
임베딩을 읽어 응용 프로그램의 조회 로직에 넘길 수 있습니다. 별도의 서버,
벡터 검색, 직렬화, 모델 재로딩 API는 없습니다. 따라서 이 패키지만으로
모델을 저장해 재시작 후 안정적으로 서빙할 수는 없습니다. 운영 서빙에는
최소한 어휘 토큰의 **바이트 배열과 행 순서**, 입력 임베딩의 차원·값을 함께
저장하고, 그 자료를 로드하여 미등록 토큰을 처리하는 조회 계층이 필요합니다.
출력 임베딩은 학습 목적 함수의 매개변수이므로 일반적인 단어 벡터 조회에는
`EmbeddingKind::Input`을 사용합니다. 학습 중 스냅샷은 좌표별 원자적
읽기일 뿐 행 전체의 일관된 시점을 보장하지 않으므로, 서빙용 스냅샷은
`Trainer::train`이 끝난 뒤 생성해야 합니다.

```rust
use std::path::Path;
use w2v::{
    Corpus, EmbeddingKind, Model, Status, Trainer, TrainingConfig,
    Vocabulary, VocabularyConfig,
};

fn train_and_get_vector(corpus_path: &Path, token: &[u8])
    -> Result<Option<Vec<f32>>, Status>
{
    let corpus = Corpus::create(corpus_path)?;
    let vocab = Vocabulary::build(&corpus, &VocabularyConfig {
        initial_capacity: 1000,
        hash_capacity: 100_003,
        min_count: 1, // 작은 예제에서 단어를 제거하지 않음
    })?;
    let config = TrainingConfig {
        thread_count: 1,
        negative_table_size: 257, // 작은 예제용; 기본값은 100,000,000
        ..TrainingConfig::default()
    };
    let model = Model::create(
        &vocab, config.embedding_dimension, config.root_seed,
        config.rng_algorithm,
    )?;
    let trainer = Trainer::create(&corpus, &vocab, &model, &config)?;
    trainer.train()?;

    let mut values = vec![0.0; model.vocab_size * model.embedding_dimension];
    if model.snapshot_into(EmbeddingKind::Input, &mut values) != Status::Ok {
        return Err(Status::InvalidArgument);
    }
    let Some(index) = vocab.find(token) else { return Ok(None) };
    let start = index * model.embedding_dimension;
    Ok(Some(values[start..start + model.embedding_dimension].to_vec()))
}
```

코드의 `corpus_path`는 호출자가 주입하는 실제 코퍼스 경로입니다. 기본
`VocabularyConfig`의 해시 테이블과 기본 negative table은 큰 데이터셋을
위한 크기이므로 작은 입력이나 제한된 메모리 환경에서는 크기를 조정해야
합니다. 예제의 작은 negative table은 학습 동작 확인용이며 품질 기준이
아닙니다.

## 오브젝티브 선택과 서브샘플링

`TrainingConfig::objective_kind`에 `ObjectiveKind::HierarchicalSoftmax` 또는
`ObjectiveKind::NegativeSampling`을 지정합니다. 기본값은 negative sampling,
음성 샘플 수는 5입니다. `Trainer::create`는 선택한 종류에 따라 negative
table을 만들거나 생략하고, 각 학습 스텝은 **선택된 오브젝티브 하나만**
실행합니다. HS는 어휘의 Huffman 경로를 사용하며
`hs_out_of_range_policy`를 적용합니다. Negative sampling은 빈도^0.75
테이블을 사용하고, 양성 타깃과 같은 음성 샘플은 재추첨 없이 건너뜁니다.
`negative_sample_count`·`negative_table_size`는 negative sampling을
선택했을 때만 학습 동작에 영향을 줍니다. 오브젝티브를 바꾸려면 새 설정으로
`Trainer`를 다시 만들고 새 학습 실행을 시작해야 합니다.

```rust
use w2v::{ModelKind, ObjectiveKind, TrainingConfig};

let mut config = TrainingConfig::for_model(ModelKind::SkipGram);
config.objective_kind = ObjectiveKind::HierarchicalSoftmax;
config.subsampling_threshold = 0.0; // 비활성화; 기본값 1e-3은 활성화
// 이 설정으로 Model과 Trainer를 생성한 뒤 Trainer::train()을 실행합니다.
```

Negative sampling을 선택하려면 `objective_kind`를
`ObjectiveKind::NegativeSampling`으로 지정하고 `negative_sample_count`를
1 이상으로 설정합니다. HS를 선택한 경우 음성 샘플 수를 0으로 바꾸어
“negative를 끄는” 방식은 사용하지 않습니다. `objective_kind`가 선택의
기준입니다.

## Training observations

`TrainingConfig::observation_interval`은 worker별 model-objective 호출 중 하나를
몇 호출마다 측정할지 정합니다. 기본값 `0`은 관측을 비활성화하며 학습 수치 경로에
추가 연산을 넣지 않습니다. 양수이면 `EpochReport::observations`에 epoch, global
processed tokens, learning rate, sampled objective loss sum/count, elapsed time 및
throughput이 기록됩니다. `EpochReport`의 loss sum/count는 해당 epoch의 dense
observation 집계입니다.

Python의 `EpochReport.observations()`도 같은 owned snapshot을 반환합니다. callback은
완료된 epoch report를 받으므로 Python 코드가 실행되는 동안 Rust 학습 메모리를
빌리지 않습니다.

## Shared update strategy

`TrainingConfig::update_strategy`의 기본값은 `UpdateStrategy::Hogwild`입니다.
Hogwild는 `AtomicU32` 저장소에서 relaxed load와 relaxed store를 사용해 CAS
재시도 없이 갱신합니다. 따라서 plain concurrent `f32` write의 data race는
없지만 worker 간 update 순서와 multi-thread 결과의 bitwise 재현성은 보장하지
않습니다. `UpdateStrategy::AtomicCas`는 coordinate-level compare-exchange가
필요한 checked/reference 비교용 선택지입니다.

이 선택은 원본 `z_original_w2v.c`의 synchronization-free Hogwild 의미와 성능을
비교하기 위한 것이며, 원본과 모듈식 C/Rust 사이의 알려진 Skip-gram, RNG,
objective 차이를 semantic equivalence로 없애려는 규칙이 아닙니다.

같은 `Trainer`에서 `train()`을 다시 호출하면 처리 토큰 카운터는 0으로
초기화되지만 이미 학습된 임베딩은 초기화되지 않습니다. 처음부터 다시
학습하려면 새 `Model`을 생성해야 합니다.

`TrainingConfig::subsampling_threshold`는 학습 중 **문장 버퍼에 넣기 전**
어휘에 있는 비경계 토큰에 적용됩니다. 기본값 `1e-3`은 활성화이며 `0.0`은
비활성화입니다. 어휘 구축이나 어휘의 빈도 집계 자체에는 적용되지
않습니다. 유지 확률은 원본의
`(sqrt(count / (sample * train_words)) + 1) * (sample * train_words) / count`
식을 따릅니다. 여기서 `train_words`는 필터링을 마친 어휘의 누적 빈도입니다.
문장 경계 `</s>`는 버퍼에 넣지 않고 문장을 끝내며, 서브샘플링되지
않습니다. 서브샘플링으로 버려진 토큰도 인식된 토큰 수와 학습률 진행률에는
계산됩니다. 따라서 `processed_tokens()`는 실제 모델 스텝 수나 유지된
문장 토큰 수가 아닙니다.

CBOW와 Skip-gram은 `TrainingConfig::for_model(ModelKind::Cbow | ModelKind::SkipGram)`로
선택할 수 있습니다. 이 생성자는 시작 학습률을 각각 0.05와 0.025로
설정합니다. `TrainingConfig::default()`에서 `model_kind`만 Skip-gram으로
바꾸면 CBOW의 기본 시작 학습률 0.05가 그대로 남으므로, 모델별 기본값을
원할 때는 `for_model`을 사용하세요.

## 원본과의 주요 차이

| 항목 | 원본 `z_original_w2v.c` | 모듈식 C와 Rust |
|---|---|---|
| Skip-gram 방향 | 문맥 단어의 입력 벡터로 중심 단어를 예측 | 중심 단어의 입력 벡터로 문맥 단어를 예측 |
| 오브젝티브 | HS와 negative를 동시에 켤 수 있고 출력 행렬 `syn1`/`syn1neg`를 분리 | 둘 중 하나를 선택하고 해당 학습의 출력 행렬 하나를 사용 |
| 난수 | 모델 초기화에 한 상태를 사용하고, 학습 워커 안에서는 창/서브샘플링/negative가 한 상태를 공유 | root seed·워커·용도별 독립 스트림; LCG 또는 Xorshift 선택 가능 |
| 병렬 업데이트와 학습률 | 공유 float/학습률을 동기화 없이 갱신 | 임베딩 좌표·처리 토큰 수는 relaxed atomic; 학습률은 워커 로컬에서 정해진 간격 후 갱신 |
| 입력·출력 | 원본 CLI, 어휘 파일, 벡터 파일, 클래스 출력 | 라이브러리 API와 메모리 내 스냅샷; 파일 포맷/서비스 미구현 |

원본과 모듈식 C의 기타 의도적 차이는
[`reference/ALGORITHM.md`](reference/ALGORITHM.md)에 기록돼 있습니다.
바이트 기반 토크나이저는 UTF-8을 요구하지 않고 줄바꿈을 `</s>`로
표현합니다. 최소 빈도·해시 테이블 70% 초과 시 가지치기·동률 정렬
결과는 어휘 행 순서에 영향을 줍니다. 재현성 확인에는 **같은 어휘 행 순서**가
필수입니다. Negative table이 문장 경계 항목 0을 뽑으면 다른 비경계
어휘 항목으로 대체하고, 양성 타깃과 같아진 음성 샘플은 재추첨하지
않습니다. 병렬 학습은 좌표별 갱신의 데이터 레이스를 없앴지만 워커 간
연산 순서는 고정되지 않아 결과의 비트 단위 재현성을 보장하지 않습니다.

## 검증

`tests/stage1_contract.rs`부터 `tests/stage6_trainer.rs`까지 C의 상수,
토크나이저, 난수, 어휘/Huffman, 네 가지 모델·오브젝티브 1-스텝 조합,
학습률 갱신과 서브샘플링 활성/비활성 경로를 검사합니다. 2-epoch 단일 스레드 학습은 CBOW/Skip-gram ×
HS/negative의 네 경우와 Xorshift CBOW/negative의 입력·출력 임베딩 해시가
모듈식 C 테스트와 일치합니다. 병렬 테스트는 학습 완료와 유한한 입력
임베딩을 확인합니다. 이는 지원 플랫폼의 작은 입력에 대한 동작 검증이며
서비스 성능, 대규모 코퍼스 품질, 원본 파일 포맷 호환성의 검증은 아닙니다.

이 디렉터리에서 `cargo test`, `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, `just check`를 실행합니다.
`just check`는 패키지 로컬 C 테스트만 실행합니다. 원본 스냅샷은 테스트,
컴파일, 포맷, sanitizer 대상에서 제외합니다.
