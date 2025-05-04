---
title: "Gradient Descent"
date: 2025-04-30 
categories: [Study, Deep Learning]
tags: [gradient descent, learning rate, partial differnetiation]  # TAG names should always be lowercase
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

## 편미분, Partial Differentiation
두 개 이상의 변수에 대한 미분을 편미분이라고 한다. 간단하게 아래와 같은 식을 예로 들수 있다.

![eq4.6](/assets/img/gradient_descent/eq4.6.png)

이 함수를 그림으로 그려보자.

![fig4-8](/assets/img/gradient_descent/fig4-8.png){: w="400", h="400"}

이 함수를 미분하기 위해서는 **어떤 변수에 대해 미분할 것인지**를 구별해야한다. 이처럼 변수가 여러개인 함수에 대한 미분을 편미분이라고 하고 원함수를 편미분한 함수를 편도함수하고 한다. 수식으로는 아래와 같이 표현한다. 

![eq1](/assets/img/gradient_descent/eq1.png)


다시 말해 각 축(변수)에 대한 방향 미분을 편미분이라고 한다. 방향 미분 축에 해당하는 변수로 나머지 변수는 상수처럼 취급하고 일변수 미분을 한 계산 결과와 동일하다. 특정 좌표에서 편미분은 그 부분의 기울기를 구한다. 
![fig2](/assets/img/gradient_descent/fig2.png){: w="400", h="400"}

## Gradient 
편도함수를 변수 순서로 묶어 놓은 벡터를 gradient라고 한다.

![fig3](/assets/img/gradient_descent/fig3.png)

모든 방향으로의 미분계수는 Gradient 와 방향을 내적하여 구할 수 있다. 이를 활용하여 특정 점에서의 접평면의 방정식을 구할 수 있다. 그런데 이 **기울기**가 의미하는 것은 뭘까? 함수 $f(x,y)=x^2+y^2$에 대해 모든 점들에서의 기울기 벡터에 (-)를 붙인 벡터를 그려본 결과는 아래 그림과 같다. 

![fig4-9](/assets/img/gradient_descent/fig4-9.png){: w="400", h="400"}

원함수의 그림과 비교해보면 함수의 기울기의 음수값은 가장 낮은 장소(최솟값)을 향하는 것처럼 보인다. 또한 가장 낮은 곳에서 멀수록 화살표의 크기가 크다. 

> 정리하자면 기울기가 가리키는 방향은 **각 위치에서 함수 값을 가장 크게 줄이는 방향**이다!!
{: .prompt-tip}

따라서 함수 𝑓와 점 𝐱는 고정되어 있고 방향의 크기는 1이라고 가정한다면

![fig4](/assets/img/gradient_descent/fig4.png)

* 방향 미분이 가장 커지는 방향은 gradient 방향이고 값은 gradient의 크기이다.
* 방향 미분이 가장 작아지는 방향은 gradient 반대 방향이고 값은 마이너스 gradient의 크기이다
* 방향 미분이 0이 되는 방향은 gradient와 수직인 방향이다.

조금 더 직관적으로, 등위선과 등위면을 생각해보자. 함수의 출력값이 같은 점들의 집합을 등위선(or 등위면)이라고 한다. 함수 $f$의 등위선(면)과 gradient 는 항상 수직이고 

![fig5](/assets/img/gradient_descent/fig5.png)

가장 빨리 올라가거나 내려가는 궤적은(즉, 함수의 최대 혹은 최소점에 가장 빨리 가까워지는 경로) 등위선에 수직인 궤적이다. 

![fig6](/assets/img/gradient_descent/fig6.png)


## Gradient Descent, 경사하강법
Machine Learning 의 목표는 Loss Function을 최소화하는 최적의 매개변수(Parameter) 값들을 찾아내는 것이다. 하지만 일반적인 경우, Loss Function은 매우 복잡하고 어떤 곳이 최소가 되는 점인지 알기 어렵다. 이런 상황에서 기울기를 이용해 함수의 최솟값을 찾으려는 것이 경사하강법이다. 

> 주의할 점은 각 지점에서 함수의 값을 낮추는 방향의 지표로 제시하는 것이 **기울기**라는 점이다. 하지만 기울기가 가리키는 방향에 실제로 함수의 최솟값이 있는지 확실하게 보장할 수는 없다. 예를 들어 **Local Minimum**, **Plateau** 혹은 **Saddle Point**에서는 기울기가 0에 가깝지만 실제로 함수의 최솟값이 아닌 경우가 있다.\
> 이러한 문제를 해결하기 위해 이후 추가적인 Optimization 기법들이 제안된다.
{: .prompt-tip}

{% include embed/youtube.html id='JdeemZDr-hU' %}

경사하강법을 한마디로 정의하자면 **Gradient 의 반대 방향으로 한 발자국씩 내딛으며 함수값을 낮추는 방법**이다. 이때 위에서 언급했듯이 다변수 함수 $f$가 가장 빠르게 감소하는 방향은 Gradient의 반대방향  $−𝛻𝑓(𝐱)$이다. 식으로 나타내면 아래와 같다. 

![eq2](/assets/img/gradient_descent/eq2.png)

현 위치에서 기울어진 방향으로 일정거리만큼 이동한다, 그렇게 이동한 새로운 위치에서 다시 기울기를 구하고 그 방향으로 다시 나아가는 과정을 반복한다. **어느 방향으로 나아갈 것인지**는 **Gradient**가 결정하고 **얼만큼 이동**할 것인지는 **Learning Rate, 학습률**이 결정한다. 

**학습률, Learning rate** 이란 한번에 얼만큼 이동할 것인지 갱신하는 양을 나타낸다. 이때 학습률은 HyperParameter로 사전에 적절한 값으로 미리 정의해두어야 한다. 만약 학습률이 너무 작거나 너무 크다면 신경망은 최적의 장소를 효율적으로 찾아갈 수 없게 된다. 따라서 일반적으로 실험을 통해 적절한 학습률을 선정한다. 일반적으로는 0.01이나 0.001에서 시작해서 점진적으로 조정해 나가는 방식을 사용한다. 

![fig7](/assets/img/gradient_descent/fig7.png)

### 경사하강법의 문제
그런데 경사하강법이 가지는 두가지 문제가 있다. 우선 계산 속도가 느리다는 단점이 있다. 일반적인 Gradient Descent의 수식에서는 모든 데이터를 고려한다. 데이터의 수가 적으면 큰 문제가 되지 않지만 만약 데이터가 매우 많아진다면 최소점에 수렴하기까지 매우 오랜 시간일 걸릴 것이다. 모든 데이터를 고려하기 때문에 신중한 방향 설정은 가능하지만 극단적인 속도 저하로 인해 현실적인 적용은 한계가 있다.

다음으로 발생할 수 있는 문제는 좋지 않는 Local Minimum에 빠질 수 있다는 점이다. 우리의 목적은 Loss function의 Global Minimum 값을 찾는 것이다. 하지만 Local Minimum 이나 Saddle Point에서는 $𝛻𝑓(𝐱_n)$의 크기가 너무 작아지고 보폭도 줄어들어 Global Minimum이 아닌 곳에 안착해버리는 문제가 발생할 수 있다. 일반적으로 더 복잡한 모델을 사용할 수록 Loss Function은 더 많은 Local Minimum point를 가지게 된다. 

![fig8](/assets/img/gradient_descent/fig8.png)

이러한 문제를 해결하기 위해 새로운 알고리즘들이 등장한다. 각각의 구체적인 알고리즘에 대해 알아보기 전에 Weight Initialization을 먼저 알아보자. 

### Weight Initialization
위에서 언급했듯이 **함수의 Gloabal Minimum에 도달하기 위해서는 시작점이 중요하다.** 초깃값을 어떻게 설정하는지에 따라 신경망의 학습 성능이 크게 달라질 수 있다. 가장 널리 알려진 방식으로는 LeCun, He, Xavier 방식이 있다. 이 세 방식들은 공통적으로 Weight를 평균이 0인 랜덤한 값으로 초기화하며 웨이트의 분산 값은 각 방식마다 다르게 설정한다. (Bias를 일반적으로 0으로 초기화하거나 작은 양수 값으로 설정한다.)

1. LeCun Initialization
   $$
   w \sim \mathcal{U}\left(-\sqrt{\frac{3}{N_\text{in}}}, \sqrt{\frac{3}{N_\text{in}}}\right)
   $$
   또는
   $$
   w \sim \mathcal{N}\left(0, \frac{1}{N_\text{in}}\right)
   $$
   * 평균은 0이고 분산은 $\frac{1}{N_\text{in}}$ 인 분포이다.
   * 가우시안 분포는 0 주변에 더 집중된 값을 선택한다.
     
2. He Initialization
   $$
   w \sim \mathcal{U}\left(-\sqrt{\frac{6}{N_\text{in}}}, \sqrt{\frac{6}{N_\text{in}}}\right)
   $$
   또는
   $$
   w \sim \mathcal{N}\left(0, \frac{2}{N_\text{in}}\right)
   $$
   * 평균은 0이고 분산은 $\frac{2}{N_\text{in}}$ 인 분포이다.
   * ReLU 활성화 함수를 사용하는 신경망에서 효과적인 것으로 알려져 있다.
     
3. Xavier Initialization
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

> $N_\text{in}$과 $N_\text{out}$ 은 각 레이어의 입력과 출력 노드의 수를 의미한다. $N_\text{in}$이 크다면 그만큼 많은 입력값이 Weight와 곱해지고 더해지기 때문에 Activation에 들어오는 값들의 분산값이 커진다. 하지만 분산이 지나치게 큰 경우, 특히 Sigmoid와 같은 함수를 사용한다면 Gradient 가 0에 가까워져 학습이 느려지는 문제가 발생할 수 있다. **이를 Gradient Vanishing 문제**라고 한다.
> 
> ![fig10](/assets/img/gradient_descent/fig10.png){: w="400", h="300"}
{: .prompt-info}

이러한 문제를 해결하기 위해 3가지 방법 모두 $N_\text{in}$이 클수록 분산을 작게 하여 Weight를 0에 더 가깝게 초기화한다. 이는 순전파 과정에서 분산이 층을 지나며 급격히 커지는 문제를 방지한다. 다만 Xavier 방식에서는 $N_\text{out}$도 함께 고려하는데, 이는 역전파 과정에서 발생하는 **Gradient Exploding 문제**를 막기 위함이다. $N_\text{out}$이 크다면 $N_\text{in}$이 클때와 마찬가지로 Gradient의 분산 값이 매우 커지게 되고 이는 학습을 불안정하게 만들 수 있다. 

결론적으로 이러한 초기화 방식들은 신경망의 깊이가 깊어져도 안정적으로 학습이 진행되도록 도와준다. 적절한 초기화는 Gradient Vanishing/Exploding 문제를 해결하고 빠르고 안정적인 학습을 가능하게 한다.

### Stochastic Gradient Descent(SGD), 확률적 경사 하강법
확률적 경사 하강법은 경사하강법의 한계를 개선한 방법이다. 경사하강법이 모든 데이터를 고려하여 Loss를 계산하는 반면, **SGD는 랜덤하게 데이터 하나를 선택하여 Loss를 계산한다.** 이처럼 무작위로 데이터를 선택하는 과정 때문에 Stochastic 이라는 이름이 붙는다. 

![fig9](/assets/img/gradient_descent/fig9.png){: w="400", h="300"}

위 그림처럼 SGD는 GD 보다 덜 신중하게 방향을 설정하지만 그렇게 때문에 더 빠르게 최소점에 도달할 수 있다. 이때 그림처럼 지그재그 방향으로 나아가는 이유는 각 시점마다 모든 데이터가 아닌 일부 데이터만을 고려한 기울기를 계산하기 때문이다. 전체적으로 보면 오히려 돌아가는 것처럼 보여서 GD보다 느리지 않을까 싶지만 한번의 업데이트를 하는데 걸리는 시간인 GD에 비해 월등히 짧기 때문에 더 빠르게 최저점에 도달할 수 있다고 한다. 

이처럼 불규칙한 움직임을 갖는다는 특성 때문에 SGD는 종종 Local Minimum을 탈출할 기회를 얻을 수 있다. 따라서 복잡한 Loss 지형에서 더 유연한 탐색을 가능하게 한다. 

{% include embed/youtube.html id='VbYTp0CIJkY' %}

### Mini-Batch Gradient Descent, 미니배치 경사 하강법
하지만 SGD는 하나의 데이터만을 고려하기 때문에 대규모 데이터셋에서는 지나치게 편향된 기울기를 계산할 수 있다는 한계가 존재한다. 

따라서 **Mini-Batch GD에서는 하나의 데이터가 아닌 여러개의 데이터 묶음을 Loss 계산에 사용한다.**\
예를 들어 Batch Size가 N이라고 한다면 N개의 데이터를 랜덤하게 뽑아 그 평균을 Loss로 삼고 gradient를 계산한다. 특히 병렬 연산을 지원하는 GPU를 활용하면 Mini Batch GD의 효율성은 훨씬 높아진다. 

![fig11](/assets/img/gradient_descent/fig11.png){: w="400", h="300"}

하지만 무작정 batch size를 키운다면 오히려 GD와 비슷해져 또 다시 Local Minimum에 빠지는 문제가 발생할 수 있기 때문에 학습 속도와 최적화 성능 사이의 Tradeoff를 고려해서 적절한 균형점을 찾아야한다. 

> batch size와 Learning Rate의 조절을 위한 흥미로운 연구가 있다고 한다[^1]. 일반적으로 Batch size가 증가할수록 Validation Error가 증가한다. 이러한 문제를 해결하기 위해 다음 두가지 방법을 제안한다. 
> 1. Linear Scaling Rule: batch size를 키울때 Learning Rate 도 비례해서 키운다.
> 2. Learning Rate Warmup: 학습 초기에 Learning Rate을 0에서 시작해서 점진적으로 증가한다.
{: .prompt-tip}

### Momentum
이름에서 알 수 있듯, 관성의 성질을 이용하는 방법이다. 이전의 GD, SGD, mini-batch GD 의 경우 현재 시점의 Gradient 만 고려한다. 하지만 이런 방법들은 특정한 상황, 특히 Loss Function이 타원형과 같은 형태일 때 문제가 발생할 수 있다. Gradient는 가장 가파른 방향을 향하기 때문에 항상 등고선에 수직하다. 따라서 타원형의 Loss Function의 경우 아래 그림과 같이 지그재그 경로로 수렴하게 된다. 

![fig12](/assets/img/gradient_descent/fig12.png){: w="500", h="400"}

이처럼 진동하면서 수렴하는 방식은 상당히 비효율적이므로 우리는 이 진동의 폭을 줄이고자 한다. 따라서 이런 문제를 해결하기 위해 **Momentum은 이전 Gradient들을 누적하여 현재의 이동 방향을 결정한다.** 

{% include embed/youtube.html id='qfb2ezDWGIU' %}

Momentum의 동작원리를 알아보자.
1. 초기에는 GD와 동일한 방향으로 이동한다.
2. 현재와 이전 Gradient를 합산하여 방향을 결정한다. 예를 들어 첫 이동이 왼쪽이었다면, 두번째 이동이 오른쪽이어도 첫 이동의 영향으로 인해 오른쪽으로의 이동이 줄어든다.
3. 세번째 이동이 왼쪽이어도 직전의 오른쪽 이동 관성으로 인해 왼쪽으로의 이동이 줄어든다.
4. 이런 과정이 반복되면서 불필요한 진동을 상쇄하며 최소점을 향해 수렴한다.

종종 관성으로 인해 최저점을 지나치는 경우도 있지만 전반적으로 더 빠르고 효율적인 학습이 가능하다. 


### RMSProp(Root Mean Squared Propagation)
Momentum이 이전의 Gradient 값들을 누적해서 더했다면, RMSProp은 각 파라미터에 대한 편미분 값을 제곱해서 누적하는 방식이다. 이는 각 파라미터들의 전체적인 이력변화를 반영하여 학습의 안정성을 높이고자 하는 시도이다.

[To be Updated...]

### ADAM(Adaptive Momentum Estimation)
현재 가장 많이 사용되는 알고리즘으로 Momentum과 RMSprop을 합친 알고리즘이다. 이전 gradient의 관성을 사용하고, learning rate의 경향을 반영한다.

[To be Updated...]



[^1]: [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017](https://arxiv.org/abs/1706.02677)

