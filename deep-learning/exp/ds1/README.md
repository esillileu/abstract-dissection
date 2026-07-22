# DS1 실행 catalog

DS1은 기존 domain CLI를 그대로 사용한다.

```bash
just exp ds1 plan -e 01 -seed 1,2,3,4
just exp ds1 run -e 01 -seed 1,2,3,4
```

`-e`는 `e01`–`e08` 형식의 catalog experiment ID를 선택한다. `-seed`/`--seed`는
`config/seeds.yaml`의 **seed registry 인덱스**다. 예를 들어 `1,2,3,4`는 두 번째부터
다섯 번째 registry seed를 선택한다.

Catalog mapping은 `e01=GT01`부터 `e08=GT08`까지다. 각 YAML의 `variants`가 실행할
atomic trial을, `policy.seed_count`가 사용할 registry seed 수를 결정한다. 따라서
`just exp ds1 plan --all -seed 0`은 첫 번째 registry seed의 49개 atomic trial을 보인다.
