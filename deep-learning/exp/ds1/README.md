# DS1 실행 catalog

DS1은 기존 domain CLI를 그대로 사용한다.

```bash
just exp ds1 plan -e 01 -seed 1,2,3,4
just exp ds1 run -e 01 -seed 1,2,3,4
```

`-e`는 `e01`–`e08` 형식의 catalog experiment ID를 선택한다. `-seed`/`--seed`는
`config/seeds.yaml`의 **seed registry 인덱스**다. 예를 들어 `1,2,3,4`는 두 번째부터
다섯 번째 registry seed를 선택한다.

특정 atomic run만 실행하려면 `-a`/`--atomic-run`, 특정 atomic run을 빼려면
`-x`/`--exclude-atomic-run`을 사용한다. 두 옵션은 함께 사용할 수 없으며, 반복하거나
쉼표로 여러 ID를 지정할 수 있다. ID는 선택한 `-e`/`--all` 범위 안에서 검증된다.

```bash
just exp ds1 plan -e 01 -a MLP-OPT-SGD -seed 0
just exp ds1 run -e 01 -a MLP-OPT-SGD,MLP-OPT-ADAM -seed 0
just exp ds1 run -e 01 -x MLP-OPT-SGD -seed 0
```

기본 실행 순서는 atomic run 우선이다. 선택한 모든 atomic run을 같은 seed끼리 먼저
실행하려면 `--seed-first`를 추가한다. 여러 experiment를 선택해도 전체 plan에
적용되며, `-seed`에 지정한 인덱스 순서를 따른다.

```bash
just exp ds1 plan -e 01-02 -seed 0-2 --seed-first
just exp ds1 run -e 01-02 -seed 0-2 --seed-first
```

Catalog mapping은 `e01=GT01`부터 `e08=GT08`까지다. 각 YAML의 `variants`가 실행할
atomic trial을, `policy.seed_count`가 사용할 registry seed 수를 결정한다. 따라서
`just exp ds1 plan --all -seed 0`은 첫 번째 registry seed의 49개 atomic trial을 보인다.
