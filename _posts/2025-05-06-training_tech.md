---
title: "Training Techniques"
date: 2025-05-06
categories: [Study, Deep Learning]
tags: [vanishing gradient, weight initialization, batch normalization, over fitting, under fitting, regularization, dropout, hyperparameter]  # TAG names should always be lowercase
# description: 

math: true

# > tip, info, warning...
# {: .prompt-tip}

# ```
# def step_function(x):
#     y=x>0
#     return y.astype(np.int)
# ```
---

이번 포스팅에서는 학습의 효율과 정확도를 높일 수 있는 다양한 기법들에 대해 알아본다[^1]. 

## Weight Initialization, 가중치 초기화

* Optimization과 Gradient Descent를 참조

**함수의 Gloabal Minimum에 도달하기 위해서는 시작점이 중요하다.** 초깃값을 어떻게 설정하는지에 따라 신경망의 학습 성능이 크게 달라질 수 있다. 이번 장에서는 학습을 시작하기 전에, 가중치의 초깃값을 어떻게 잡는것이 학습에 유리한지에 대해 알아보자.

우선 가중치를 평균이 0, 표준편차가 1인 정규분포로 가중치를 기화하는 상황을 가정해보자. 활성화 함수로 sigmoid 함수를 사용하는 5층 신경망에 무작위로 생성한 입력 데이터를 넣고 얻은 활성화 결과를 확인해보자. 각 층의 활성화 분포는 아래와 같다.

![fig10](/assets/img/training_tech/fig10.png){: w="400", h="300"}

각 층의 활성화 값이 0과 1에 치우쳐서 분포하게 된다. 하지만 Sigmoid 활성화 함수의 개형을 생각해보면 그 출력이 0또는 1에 가까워지면 미분값은 0에 가까워진다. 

![fig12](/assets/img/training_tech/fig12.png){: w="400", h="300"}

따라서 데이터가 0과 1에 치우쳐서 분포하게 되면 역전파의 기울기 값이 점점 작아지다가 사라지고 이를 **Vanishing Gradient , 기울기 소실**문제라고 한다. 특히 층이 깊어질수록 기울기 소실은 큰 문제가 된다. 

![fig13](/assets/img/training_tech/fig13.png){: w="600", h="400"}


다음은 반대로 평균이 0, 표준편차가 0.01인 정규 분포로 가중치를 초기화하는 상황을 가정하자. 각 층의 활성화 분포는 아래와 같다. 

![fig11](/assets/img/training_tech/fig11.png){: w="400", h="300"}

이번에는 각 층에서 활성화 값이 0.5부근에 몰려서 분포한다. 이 경우 기울기 소실의 문제는 발생하지 않지만 **표현력을 제한**한다는 점에서 문제가 생긴다. 이런 상황에서는 여러 뉴런이 비슷한 값을 출력하므로 뉴런을 여러개 가지는 의미가 없어진다. 

> 따라서 각 층의 활성화 값은 적당히 고르게 분포되어야 한다. 층과 층 사이에 다양한 데이터가 흘러야 신경망의 학습이 효율적으로 이루어질 수 있다. 
{: .prompt-info}

그렇다면 어떤 방식으로 가중치의 초깃값을 **적절히** 분포시킬 수 있을까?? 가장 널리 알려진 방식으로는 LeCun, He, Xavier 방식이 있다. 이 세 방식들은 공통적으로 Weight를 평균이 0인 랜덤한 값으로 초기화하며 웨이트의 분산 값은 각 방식마다 다르게 설정한다. (Bias를 일반적으로 0으로 초기화하거나 작은 양수 값으로 설정한다.)

### LeCun Initialization
$$
w \sim \mathcal{U}\left(-\sqrt{\frac{3}{N_\text{in}}}, \sqrt{\frac{3}{N_\text{in}}}\right)
$$
또는
$$
w \sim \mathcal{N}\left(0, \frac{1}{N_\text{in}}\right)
$$
* 평균은 0이고 분산은 $\frac{1}{N_\text{in}}$ 인 분포이다.
* 가우시안 분포는 0 주변에 더 집중된 값을 선택한다.

LeCun 초깃값을 이용할때 활성화 분포는 아래와 같다.

![lecun](/assets/img/training_tech/lecun.png){: w="400", h="300"}

### He Initialization
$$
w \sim \mathcal{U}\left(-\sqrt{\frac{6}{N_\text{in}}}, \sqrt{\frac{6}{N_\text{in}}}\right)
$$
또는
$$
w \sim \mathcal{N}\left(0, \frac{2}{N_\text{in}}\right)
$$
* 평균은 0이고 분산은 $\frac{2}{N_\text{in}}$ 인 분포이다.
* ReLU 활성화 함수를 사용하는 신경망에서 효과적인 것으로 알려져 있다.

He 초깃값을 이용할때 활성화 분포는 아래와 같다.

![he](/assets/img/training_tech/he.png){: w="400", h="300"}

### Xavier Initialization
$$
w \sim \mathcal{U}\left(-\sqrt{\frac{6}{N_\text{in} + N_\text{out}}}, \sqrt{\frac{6}{N_\text{in} + N_\text{out}}}\right)
$$
또는
$$
w \sim \mathcal{N}\left(0, \frac{2}{N_\text{in} + N_\text{out}}\right)
$$
* 평균은 0이고 분산은 $\frac{2}{N_\text{in} + N_\text{out}}$ 인 분포이다.
* 다른 초기화 방식들과 달리 $ N_\text{out}$도 고려한다.
* 다른 방식들보다 작은 분산으로 0에 더 가깝게 초기화 된다.
* Sigmoid 나 tanh와 같은 활성화 함수를 사용하는 신경망에서 효과적인 것으로 알려져 있다.

Xavier 초깃값을 이용할때 활성화 분포는 아래와 같다.

![xavier](/assets/img/training_tech/xavier.png){: w="400", h="300"}

> $N_\text{in}$과 $N_\text{out}$ 은 각 레이어의 입력과 출력 노드의 수를 의미한다. $N_\text{in}$이 크다면 그만큼 많은 입력값이 Weight와 곱해지고 더해지기 때문에 Activation에 들어오는 값들의 분산값이 커진다. 하지만 분산이 지나치게 큰 경우, 특히 Sigmoid와 같은 함수를 사용한다면 Gradient 가 0에 가까워져 Gradient Vanishing 문제가 발생할 수 있다.
{: .prompt-info}

이러한 문제를 해결하기 위해 3가지 방법 모두 $N_\text{in}$이 클수록 분산을 작게 하여 Weight를 0에 더 가깝게 초기화한다. 이는 순전파 과정에서 분산이 층을 지나며 급격히 커지는 문제를 방지한다. 다만 Xavier 방식에서는 $N_\text{out}$도 함께 고려하는데, 이는 역전파 과정에서 발생하는 **Gradient Exploding 문제**를 막기 위함이다. $N_\text{out}$이 크다면 $N_\text{in}$이 클때와 마찬가지로 Gradient의 분산 값이 매우 커지게 되고 이는 학습을 불안정하게 만들 수 있다. 

결론적으로 이러한 초기화 방식들은 신경망의 깊이가 깊어져도 안정적으로 학습이 진행되도록 도와준다. 적절한 초기화는 Gradient Vanishing/Exploding 문제를 해결하고 빠르고 안정적인 학습을 가능하게 한다.

## Batch Normalization, 배치 정규화
가중치 초깃값을 적절히 설정하면 각 층의 활성화 값 분포가 적당히 퍼지면서 학습이 원활해지는 것을 확인했다. 배치 정규화에서는 **각 층이 활성화 값을 적당히 퍼뜨리도록 강제**하는 방법을 알아보자.[^2]


배치 정규화는 나온지 오래되지 않았음에도 불구하고 뛰어난 결과로 인해 여러 주목을 받는 기술이다. 배치 정규화의 의의는 
1. 학습을 빨리 진행할 수 있다.
2. 초깃값에 크게 의존하지 않는다.
3. 오버피팅을 억제한다.(DropOut 필요성감소)
등이 있다.

기본 아이디어는 **각 층의 활성화 값이 적당히 분포되도록 조정하는 것이다**. 이를 위해 아래와 같이 Batch Norm 계층을 신경망에 추가한다. 

![fig6-16](/assets/img/training_tech/fig6-16.png){: w="600", h="400"}

여기서 **배치 처리**란, 여러개의 입력 데이터를 묶어서 한번에 처리하는 방식을 의미한다. 특히 GPU를 활용하는 대부분의 수치 계산 라이브러리들은 작은 배열을 한개식 처리할때 보다 여러개의 배열을 한번에 처리할때 훨씬 효율적으로 동작한다고 한다. 우선 자주 나오는 용어를 간단히 정리하자.
* 배치: 묶음
* 전체 데이터의 수 : N
* 배치 크기 : BS
* 배치 수 : N/BS
* 1 Iteration: 일반적으로 하나의 batch에 대해서 가중치를 업데이트 하는 것을 1 Iteration을 수행했다고 한다.
* 1 Epoch : 전체 학습 데이터를 한번 사용한 것을 1 Epoch이라고 한다. 여러 Epoch를 통해 학습을 진행한다.

배치 정규화는 학습시 미니배치 묶음으로 학습을 진행한다. 일반적으로 평균이 0, 분산이 1이 되도록 미니배치 단위로 정규화한다. 수식은 아래와 같다. 

![e6.7](/assets/img/training_tech/e6.7.png){: w="400", h="400"}

미니배치 $B=[{x_1, x_2, ... x_m}]$이라는 $m$개의 입력 데이터 집합에 대해서 평균 $\mu_B$와 $\sigma^2_B$를 구한다. 그리고 위 식을 이용해서 입력데이터의 평균과 분산이 각각 0과 1이 되도록 정규화한다. 이와 같은 처리는 활성화 함수앞에 추가하여 데이터 분포를 덜 치우치게 할 수 있다. 

또한 다음 수식을 이용해 배치 정규화 계층마다 고유한 확대(Scale)과 변환(Shift)를 수행한다. 

![e6.8](/assets/img/training_tech/e6.8.png)

처음에는 $\gamma=1, \beta=0 $부터 시작하여 학습이 진행됨에 따라 적절한 값으로 조정해나간다. 

이와 같은 과정을 배치 정규화라고 한다. 이때 Train시와 Text 시 배치 정규화 적용여부가 달라지는데, 학습이 완료된 후 Inference(Text) 단계에서는 하드웨어가 허용하는 한 가능한 큰 배치사이즈를 사용하는 것이 효율적이다. 하지만 **Train**시에는 적절한 Batch Size를 설정하고 학습을 해야 신경망의 효율적인 학습이 가능하다. 
* 만약 Batch Size가 너무 작은 경우, GPU를 효율적으로 사용하지 못하고 학습이 불안정해질 수 있다. 국소적인 부분만 고려해서 학습을 진행하기 때문이다.
* 반면 batch Size가 너무 큰 경우는 GPU 메모리 용량에 따라 학습이 불가능 할 수 있으며 한번 학습하는데 시간이 오래 걸리고 학습에 필요한 Epoch수가 많아질 수 있다. 또한 BS가 너무 크면 일반화 성능이 오히려 떨어질 수 있다는 연구 결과가 있다고 한다.

아래 그림은 MNIST 데이터에 대해 실험한 결과로, 실선은 배치 정규화를 사용한 경우, 점선은 사용하지 않은 경우이다. 각 경우마다 가중치 초깃값의 표준 편차를 달리해서 실험을 했는데 모든 경우 배치 정규화를 사용한 경우가 학습 진도가 더 빠른 것을 확인할 수 있다. 

![fig6-19](/assets/img/training_tech/fig6-19.png){: w="400", h="400"}

## Overfitting 
**Overfitting 이란 신경망이 훈련데이터에 지나치게 적응되어 그 외의 실제 데이터에 대해서는 제대로 대응하지 못하는 상태를 의미한다[^3].**

> 쉬운 예시로 Train 데이터를 "문제집", Test 데이터를 "수능"이라고 한다면 문제집만 너무 많이 풀어서 수능 문제는 잘 못푸는 경우를 생각해볼 수 있다.
{: .prompt-tip}

아래 그림과 같은 회귀 문제가 있을 때, overfit 한 모델은 Train 데이터에 대해서는 완벽한 성능을 내지만 새로운 데이터에 대해서는 오차가 오히려 클 수 있다. 이처럼 **좋은 모델**이란 성능 뿐만 아니라 **얼마나 범용적인가, 일반화가 가능한가로도 평가된다.** 따라서 우리는 주어진 데이터를 사용해서 그것을 잘 표현할 뿐만 아니라 새로운 데이터에도 적용할 수 있는 범용적인 모델을 만드는 것이 목표다. 

![fig1](/assets/img/training_tech/fig1.png)

Overfitting은 주로 다음의 두 경우에 일어난다.

* 데이터 수에 비해 Feature가 많고 표현력이 높은 모델
* Feature수에 비해 Train 데이터가 적은 경우

이와 같은 상황을 가정하고 MNIST 데이터를 학습 시킨 결과는 아래와 같다. 

![fig6-20](/assets/img/training_tech/fig6-20.png){: w="300", h="300"}

Train 데이터의 정확도는 약 100에포크부터 100%에 가까운 반면, Text 데이터는 그에 훨씬 못미치는 성능을 보인다. 즉 Train 데이터에 대해서만 모델이 과도하게 학습해버린 상황이다. 

### Regularization
Overfitting 은 Regulzation이라는 규제 기법을 사용하여 완화할 수 있다. **Regularization 이란 모델이 데이터에 과도하게 Fit 되었으니 이를 덜 적합하게 하여 일반성을 띄게 해주는 기법이다.** 
대표적인 방법으로는 **Weight Decay와 Drop Out 기법**이 있다.

#### Weight Decay, 가중치 감소
Weight Decay란 오버피팅을 억제하기 위한 규제 방법중 한가지로 학습 과정에서 큰 가중치에 대해서는 그에 상응하는 큰 Penalty를 부과해서 오버피팅을 억제하는 기법이다. 일반적으로 오버피팅은 가중치 매개변수의 값이 커서 발생하는 경우가 많다. 

* $𝐿^2-regularization : 새로운 손실함수 = 기존 손실함수 + 1/2×𝜆×(가중치의 제곱의 합)$
* $𝐿^1-regularization : 새로운 손실함수 = 기존 손실함수 + 𝜆×(가중치의 절대값의 합)$

위와 같이 손실함수에 **추가적인 Penalty 항**을 더해서 새로운 손실함수를 정의한다. 이때 가중치 제곱합을 더하는지, 절대값합을 더하는지에 따라 L2 regularization과 L1 regularization이 있다.\
Optimizer들은 손실함수 값을 낮추고 이로 인해 기존 손실함수와 Penalty로 인한 항 모두 낮아진다. 기존 손실함수는 예측값과 정답 사이의 오차이므로 이 값을 낮춘다는 것은 가중치를 Data에 맞도록 fitting 해 나가는 과정이다. 하지만 Penalty 항은 가중치의 합으로 구성되어 있고 이 값을 낮춘다는 의미는 가중치를 0에 가깝도록 조정하여 정보를 잃게 만드는 과정이다. 즉 Penalty 항은 기존 손실함수의 최적화로 인한 가중치의 과도한 데이터 피팅을 억제하는 역할을 한다. $𝜆$는 종규화 세기를 조정하는 하이퍼파라미터로 이 값이 클수록 큰 가중치에 대한 penalty가 커진다. 

L2 Regularization을 구현한 결과는 아래와 같다. 

![fig6-21](/assets/img/training_tech/fig6-21.png){: w="300", h="300"}

앞선 결과와 달리 Train 정확도와 Test 정확도의 차이가 줄어들었다. 다시 말해 오버피팅이 억제되었음을 알수 있다, 또한 Train 정확도 역시 100%보다 줄어든 것을 확인할 수 있다. 

#### Drop Out
L1, L2 regularization은 손실함수에 penalty 항을 추가하는 가중치 감소 방법이다. 하지만 신경망이 복잡해지면 이 방법은 효과적이지 않을 수 있다. 이런 경우 Drop Out이라는 기법을 추가적으로 도입할 수 있다. **Drop Out이란 훈련 시 은닉층의 뉴런을 무작위로 골라 삭제하는 방법이다.** 주의할 점은 Drop Out은 학습시에만 적용하고 Test 시에는 모든 뉴런을 사용해야 한다.[^4]

![fig6-22](/assets/img/training_tech/fig6-22.png){: w="400", h="300"}

> 뉴런의 연결을 끊는 것이 왜 더 좋은 성능을 내는걸까? 내가 전에 진행한 Reservoir Computing 연구에서 사용한 ESN모델은 다양한 하이퍼 파리미터가 있었다. 그 중 Reservoir 안의 뉴런들 간의 연결 정도를조정하는 Connectivity(0.0~1.0) 파라미터가 있다. 이때 connectivity를 낮추면 뉴런들이 Fully Connected 되지 않고 일부 뉴런들만 연결이 되는데 이렇게 될 경우 뉴런들로 연결된 각 경로가 특정 Feature에 특화된 학습만을 진행할 수 있고 이후 가중치의 조합을 통해 더 풍부한 표현이 학습 가능하다고 한다. Drop out도 아마 비슷한 맥락이지 않을까 생각한다.[^5]
{: .prompt-info}

Drop Out을 사용한 실험 결과이다. 

![fig6-23](/assets/img/training_tech/fig6-23.png){: w="600", h="400"}

drop out을 적용하면 훈련 데이터와 시험 데이터의 정확도 차이가 줄어드는 것을 확인할 수 있다. 

추가적으로 Drop Out은 **앙상블 기법**과 밀접한 관계가 있다. 앙상블기법이란 개별적으로 학습시킨 여러 모델의 출력을 평균내어 추론하는 방식이다. 이때 Drop Out에서 매범 무작위로 랜덤한 뉴런을 삭제하고 학습하는 행위는 다른 여러 모델을 학습시키는 것으로 해석할 수 있다. 그리고 추론 시 뉴런의 출력에 삭제한 비율을 곱함으로 앙상블에서 여러 모델을 평균내는 것과 같은 효과를 낼수 있다. 


## HyperParameters, Validation Data 
일반적으로 **Parameter**란 데이터를 통해 모델이 학습하는 값이다. 예를 들어 Weight, Bias, Scale & Shift Factor 등이 있다. 이는 데이터가 많을수록 모델이 학습을 통해 쉽게 최적의 값을 찾아낼 수 있다. 

반면 **HyperParameter**는 사람이 직접 결정하는 값이다. 예를 들어 은닉층 뉴런의 개수, 층의 개수, Learning rate, 손실함수 종류, optimizer 종류 등이 있다. 좋은 Hyper Parameter 값을 찾아내기 위해서는 반복적인 실험을 통해 알아낼 수 밖에 없다. 

### Validation Data 
모델 학습의 목적은 Overfitting을 피하고 새로운 데이터에 대해서도 잘 대응하는 범용적인 모델을 만드는 것이다. 그렇다면 실제 데이터에 대한 모델의 성능을 어떻게 판단할까? 아래 예시를 보자. Train Loss는 일정 시점이 지난 이후에 비교적 일정한 값으로 수렴한다. 

![fig0](assets/img/training_tech/fig0.jpg){: w="400", h="300"}

이때 언제 실험을 중단하는 것이 좋을지는 Test 데이터를 확인하면 알 수 있다. 그림처럼 Test 데이터의 Loss가 낮은 곳에서 학습을 멈추는 것이 이상적일 것이다. 하지만 Test 데이터를 기준으로 모델을 선택하게 되면 모델이 Test 데이터에 과적합 되어 실제 데이터에는 잘 대응하지 못할 수 있는 문제가 있다. 이런 경우 **Validation Data**를 사용할 수 있다. Validation data는 훈련데이터의 일부를 떼어내서 만들 수 있다. 마치 본 시험(Test Data)을 보기 전 모의고사(Validation Data)를 보는 것과 같다. 또한 Validation Data를 통해 HyperParameter를 조정할 수 있다. 추가적으로 Validation Data는 모델 구조 선택에도 활용할 수 있다. Model1과 Model2가 있을 때 model1의 Train Loss는 더 낮을지 몰라도 model2의 Validation Loss가 더 낮을 수 있다. 이처럼 Validation loss가 낮은 모델을 선택하는 것이 실제 데이터의 적용에 더 좋은 성능을 낼 가능성이 있다. 

> 예시로 Train 데이터를 "문제집", Validation 데이터를 "모의고사", Test 데이터를 "수능"이라고 생각할 수 있다. 문제집을 통해 전반적인 실력을 높이고, Validation 데이터를 통해 Hyper Parameter 값을 튜닝하고 마지막 Test 데이터로 시험을 친다.
* Train Data : 매개 변수 학습
* Validation Data : 하이퍼파라미터 성능 평가
* Test Data : 신경망의 범용적인 성능 평가 
{: .prompt-tip}

Validation Data는 어떻게 만들까? 가장 간단하게 생각할 수 있는 데이터 분할 방법은 훈련데이터 일부를 Validation data로 분할하는 방법이다. 

![fig2](/assets/img/training_tech/fig2.png){: w="400", h="300"}

하지만 데이터가 적을 경우, **K-fold Validation**과 같은 기법을 적용할 수 있다. 이는 데이터를 K-등분 한 후 하나를 검증용으로, 나머지를 훈련용으로 사용하고 전체 결과를 평균내어 사용하는 방법이다. 

![fig3](/assets/img/training_tech/fig3.png){: w="400", h="300"}



[^1]: 본 포스팅은 사이토 고키의 "밑바닥부터 시작하는 딥러닝"과 수원대학교 한경훈 교수님의 강의를 참고했습니다. 
[^2]: [배치 정규화에 대해 정리한 블로그](https://ffighting.net/deep-learning-paper-review/vision-model/batch-normalization/)
[^3]: [오버피팅과 규제에 대한 블로그](https://nanunzoey.tistory.com/entry/%EA%B3%BC%EC%A0%81%ED%95%A9Overfitting%EA%B3%BC-%EA%B7%9C%EC%A0%9CRegularization)
[^4]: [드롭아웃에 대한 블로그](https://brunch.co.kr/@donghoon0310/36)
[^5]: [ESN에 대해 참조할만한 논문](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE06267613)
