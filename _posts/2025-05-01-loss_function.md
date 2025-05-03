---
title: "Loss Function"
date: 2025-04-30 
categories: [Study, Deep Learning]
tags: [mean squred error, cross entropy error, loss function]  # TAG names should always be lowercase
description: 

math: true

# > tip, info, warning...
# {: .prompt-tip}

# ```
# def step_function(x):
#     y=x>0
#     return y.astype(np.int)
# ```
---
## Loss Function 
인간이 미리 기준을 정하고 그 기준에 맞게 분류하던 기존의 Rule Based Approach에서는 다양한 Feature를 모두 Cover하는 규칙을 모두 찾기란 불가능하다는 한계가 존재한다. 따라서 Machine Learning 에서는 machine이 스스로 데이터에서 규칙을 찾아내는 Data Driven Approach를 사용한다.[^1]

[^1]: 본 포스팅은 "밑바닥부터 시작하는 딥러닝"과 수원대학교 한경훈 교수님의 강의를 참고했습니다.

![fig1](/assets/img/loss_function/fig1.png)

이렇게 Machine이 스스로 규칙을 찾아내도록 하기 위해서는 Machine이 규칙을 잘 찾아냈는지 아닌지 판단할 수 있는 기준이 필요하다. 인공신경망은 현재의 상태를 하나의 지표로 표현한다. 그리고 그 지표를 가장 좋게 만들어주는 가중치 매개변수를 탐색하는 것이다.\
따라서 신경망은 "하나의 지표"를 **기준**으로 최적의 매개변수 값을 탐색한다. 신경망 학습에서 사용하는 지표는 **Loss Function, 손실함수**라고 한다. 현재의 신경망이 얼만큼 부정확하게 예측을 하고 있는지를 나타내는 지표이다.

즉 인공신경망에 **손실함수**라는 벌점을 주고, 인공신경망의 성능이 좋을수록 손실함수 값이 낮고, 성능이 나쁠수록 손실함수 값이 높다. Machine은 데이터를 통해 손실함수의 값을 낮추고자 한다. 

![fig2](/assets/img/loss_function/fig2.png){: w="600" , h="400"}

일반적으로 자주 사용하는 손실함수에는 **평균 제곱 오차, Mean Squared Error(MSE)** 와 **교차 엔트로피 오차, Cross Entropy Error(CSE)** 가 있다. 

### Mean Squared Error, MSE
평균 제곱 오차의 수식은 다음과 같다. 

![eq4.1](/assets/img/loss_function/eq4.1.png)

```
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```

* $y_k$는 신경망의 출력(신경망이 추정한 값)
* $t_k$는 정답 레이블, 라벨을 원 핫 인코딩한 후 𝑘번째 좌표
* $k$는 데이터의 차원 수를 나타낸다.

```
y=[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t=[0,0,1,0,0,0,0,0,0,0]
```

이때 $t$처럼 한 원소에 해당하는 인덱스만 1로 하고 나머지는 0으로 나타내는 표기법을 **One-hot-encoding**이라고 한다.\
데이터가 신경망을 거처 나온 확률벡터와 라벨을 원핫인코딩하여 나온 확률 벡터를 "고차원 공간의 점"으로 이해한 후, **피타고라스 정리**로 거리를 측정한다. 이렇게 구한 거리가 멀면 Loss가 크고, 거리가 작으면 Loss가 작다고 판단한다. 

데이터셋의 평균 제곱오차는 각 데이터의 MSE의 평균으로 정의한다. 


### Cross Entropy Error, CSE
교차엔트로피는 정보이론에서 **두 확률 분포사이의 거리를 재는 방법**이다. 데이터가 신경망을 거쳐 나온 확률 벡터와 라벨을 원핫 인코딩하여 나온 확률 벡터의 교차 엔트로피는 아래와 같다. 

![eq4.2](/assets/img/loss_function/eq4.2.png)

* $y_k$는 신경망의 출력(신경망이 추정한 값)
* $t_k$는 정답 레이블, 라벨을 원 핫 인코딩한 후 𝑘번째 좌표
* $k$는 데이터의 차원 수를 나타낸다.


```
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
```

수식상으로는 $\sum$이 있긴 하지만 원핫레이블된 $t_k$에 의해 해당 인덱스를 제외하고는 모두 0이 된다. 

아래 그림처럼 CSE 역시 예측값이 라벨과 가까울수록 0에 가깝고 예측값이 라벨과 멀면 값이 커진다. 

![fig3](/assets/img/loss_function/fig3.png){: w="400" , h="300"}
