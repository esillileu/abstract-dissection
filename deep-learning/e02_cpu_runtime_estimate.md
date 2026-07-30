# E02 CPU 원자 런 실행 시간 추정

이 문서는 `just exp ds2 analyze -e 02 -s`의 요약 형식에 맞춘 GT02 CPU
학습 시간 추정치다. 각 값은 atomic run 한 번의 10 epoch 학습 시간이며,
완료된 전체 CPU run의 실측값이 아니라 단축 benchmark를 전체 92,950 update로
외삽한 값이다.

```text
e02 CPU runtime estimate (mean ± sample standard deviation; min-max)
[W2V-PTB-CBOW-NS]
final_loss: no completed values
training_time (s): 4070.5 ± 576.5, [3561.5, 4781.8], n=5
[W2V-PTB-SKIPGRAM-NS]
final_loss: no completed values
training_time (s): 6676.0 ± 296.8, [6233.9, 7043.5], n=5
[W2V-PTB-CBOW-FULL]
final_loss: no completed values
training_time (s): 8126.3 ± 1288.0, [6663.1, 9617.2], n=5
[W2V-PTB-SKIPGRAM-FULL]
final_loss: no completed values
training_time (s): 41082.8 ± 1200.6, [39685.3, 42606.4], n=5
```

## CSV 호환 요약

`e02_summary.csv`의 training-time 행과 같은 열 및 표시 정밀도를 사용한다.
여기서 `seed_runs`는 외삽에 사용한 benchmark 반복 수다.

| series                | metric          | seed_runs | unit    |    mean | standard_deviation | minimum | maximum |
| --------------------- | --------------- | --------: | ------- | ------: | -----------------: | ------: | ------: |
| W2V-PTB-CBOW-NS       | training_time_s |         5 | seconds |  4070.5 |              576.5 |  3561.5 |  4781.8 |
| W2V-PTB-SKIPGRAM-NS   | training_time_s |         5 | seconds |  6676.0 |              296.8 |  6233.9 |  7043.5 |
| W2V-PTB-CBOW-FULL     | training_time_s |         5 | seconds |  8126.3 |             1288.0 |  6663.1 |  9617.2 |
| W2V-PTB-SKIPGRAM-FULL | training_time_s |         5 | seconds | 41082.8 |             1200.6 | 39685.3 | 42606.4 |

## 측정 조건

- 측정일: 2026-07-24
- CPU: AMD Ryzen 7 8845HS, 8 cores / 16 threads
- 수치 backend: NumPy 2.4.6, OpenBLAS 0.3.31, 16 threads
- dataset: PTB train, 929,579 context-target examples
- loader: batch size 100, drop-last
- budget: 10 epochs, 92,950 updates
- Word2Vec: window size 5, embedding size 100
- optimizer: Adam, learning rate 0.001
- negative sampling: 5 negatives, conditional target-exclusion sampler
- 반복: seed 300–304의 단축 benchmark 5회
- 외삽 전 각 조건의 측정 길이: CBOW-NS 200, Skip-gram-NS 150,
  CBOW-FULL 100, Skip-gram-FULL 40 updates
- 전체 PTB shuffle 비용은 별도 측정하여 실제 실행과 같은 epoch당 1회로
  보정했다.

이 추정치는 `timing_windows.csv`의 `train_wall_time_ns` 합에 대응하는 학습
시간을 목표로 한다. MLflow 설정, artifact 기록, 최종 checkpoint 저장 등
학습 window 밖의 실행기 부대 시간은 포함하지 않는다.
