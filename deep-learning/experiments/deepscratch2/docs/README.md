# deepscratch2 문서

이 디렉터리는 『밑바닥부터 시작하는 딥러닝 2』를 재현·확장하는 `deepscratch2` 실험 도메인의 설계·실행 문서를 소유한다. 실행 설정은 인접한 [`../config/`](../config/)에, 분석 코드는 [`../analysis/`](../analysis/)에 있다.

- [책 원본 대비 차이](difference.md)
- [MLflow metric contract](mlflow_metrics.md)

## 실험 설계

- [e01 CBOW-Skip-gram](experiments/e01_cbow_skipgram.md)
- [e02 Full softmax-Negative sampling](experiments/e02_full_softmax_negative_sampling.md)
- [e03 RNNLM 비교](experiments/e03_rnnlm_비교.md)
- [e04 Seq2seq 입력 반전과 구조 비교](experiments/e04_seq2seq_입력반전_구조.md)
- [e05 Attention Seq2seq](experiments/e05_attention_seq2seq_date.md)

## 실행 그룹

원자 실행의 구조·공통 정책은 [`execution_groups/`](execution_groups/)에 있다. 각 문서는 YAML의 `execution_group_id`와 대응한다.

상위 `experiments/docs/`에는 여러 도메인이 공유하는 MLflow 스키마·런타임 명세만 둔다.
