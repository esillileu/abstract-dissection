# DS2 profiling

프로파일 코드는 실험별 디렉터리에 둔다.

```bash
just exp profile ds2 -e 02
```

- `e02/update.py`: 실제 Word2Vec 1-update 및 학습시간 추정
- `e02/modules.py`: model/objective/optimizer 구성요소별 측정
- `e02/analyze.py`: e02 결과 비교와 보고서 생성
- `e02/README.md`: e02 실행 방법과 측정 계약

동기화, 반복 benchmark, 통계, 학습시간 외삽은
`src/mlprosection/profiling/benchmark.py`의 공통 API를 사용한다.
