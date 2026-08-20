# DS1 실행 catalog

DS1은 통합 `exp` CLI의 action/domain 순서를 사용한다.

```bash
just exp plan deepscratch ds1 -e 01 --seed 1,2,3,4
just exp run deepscratch ds1 -e 01 --seed 1,2,3,4
```

`-e`는 `e01`–`e10`과 확장 실험 `e12`–`e14`의 catalog experiment ID를 선택한다. `--seed`는
`config/seeds.yaml`에 등록된 **실제 master seed 값**이다. 기본 `research_v1`에서는
`1`부터 `10`까지를 사용한다.

특정 atomic run만 실행하려면 `-a`/`--atomic-run`, 특정 atomic run을 빼려면
`-x`/`--exclude-atomic-run`을 사용한다. 두 옵션은 함께 사용할 수 없으며, 반복하거나
쉼표로 여러 ID를 지정할 수 있다. ID는 선택한 `-e`/`--all` 범위 안에서 검증된다.

```bash
just exp plan deepscratch ds1 -e 01 -a MLP-OPT-SGD --seed 1
just exp run deepscratch ds1 -e 01 -a MLP-OPT-SGD,MLP-OPT-ADAM --seed 1
just exp run deepscratch ds1 -e 01 -x MLP-OPT-SGD --seed 1
```

기본 실행 순서는 atomic run 우선이다. 선택한 모든 atomic run을 같은 seed끼리 먼저
실행하려면 `--order seed-first`를 추가한다. 여러 experiment를 선택해도 전체 plan에
적용되며, `--seed`에 지정한 값의 순서를 따른다.

```bash
just exp plan deepscratch ds1 -e 01-02 --seed 1-3 --order seed-first
just exp run deepscratch ds1 -e 01-02 --seed 1-3 --order seed-first
```

Catalog mapping은 `e01=GT01`부터 `e08=GT08`까지이며, `e12=GT09`, `e13=GT10`,
`e14=GO03`이다. e13은 교재의 two-layer net backprop 학습이고 e14는 교재의
numerical/backprop gradient check다. e12는 교재 옵션을
모두 적용한 MLP와 기존 GT07 DeepConvNet을 비교하는 확장 실험이다. 각 YAML의 `variants`가 실행할
atomic trial을, `policy.seed_count`가 사용할 registry seed 수를 결정한다. 따라서
`just exp plan deepscratch ds1 --all --seed 1`은 master seed 1의 68개 atomic trial을 보인다.
