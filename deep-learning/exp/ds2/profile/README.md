# DS2 profiling

프로파일 코드는 실험별 디렉터리에 둔다.

```bash
just exp profile ds2 -e 02
just exp profile ds2 -e 05
```

- `e02/update.py`: 실제 Word2Vec 1-update 및 학습시간 추정
- `e02/modules.py`: model/objective/optimizer 구성요소별 측정
- `e02/analyze.py`: e02 결과 비교와 보고서 생성
- `e02/README.md`: e02 실행 방법과 측정 계약
- `e05/benchmark.py`: BetterRnnlm full update 및 TimeLSTM before/after 측정
- `e05/validation.py`: Phase 1 정확성, lockstep, 재현성 gate
- `e05/run_nsys.sh`: e05 NVTX/CUDA Nsight trace와 요약
- `e05/report.md`: 추적되는 Phase 1 결과와 중간 정지점

동기화, 반복 benchmark, 통계, 학습시간 외삽은
`src/mlprosection/profiling/benchmark.py`의 공통 API를 사용한다.
