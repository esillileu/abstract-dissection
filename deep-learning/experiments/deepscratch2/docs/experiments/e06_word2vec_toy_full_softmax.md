# e06. Word2Vec toy full softmax

## 원본

『밑바닥부터 시작하는 딥러닝 2』 원본 저장소 [`deep-learning-from-scratch-2`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/)의 [`ch03/simple_cbow.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch03/simple_cbow.py)와 [`ch03/train.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch03/train.py)를 직접 재현한다. [`ch03/simple_skip_gram.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch03/simple_skip_gram.py)의 같은 toy 조건 비교는 도메인에서 추가한 보조 확장이다.

책 ch03의 toy 문장 `You say goodbye and I say hello.`에서 full softmax Word2Vec을 학습한다.

| 항목 | 내용 |
|---|---|
| 독립변인 | CBOW, Skip-gram |
| 고정변수 | window 1, embedding 5, batch 3, Adam .001, 1,000 epochs |
| 목적 | ch03의 CBOW baseline을 보존하고 같은 toy 조건의 Skip-gram을 보조 비교한다. |
| 관찰값 | prediction-term normalized train loss, 최종 embedding checkpoint |

## 조회할 원자 실행

`W2V-TOY-CBOW-FULL`, `W2V-TOY-SG-FULL`

## 해석 제한

CBOW는 ch03의 책 조건이고, Skip-gram은 같은 모델 코드에 적용한 확장 조건이다.
