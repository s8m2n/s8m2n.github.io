---
title: "Weight Initialization"
date: 2025-05-06
categories: [Study, Deep Learning]
tags: [vanishing gradient, weight initialization]  # TAG names should always be lowercase
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

## Weight Initialization[^1]

* Optimization과 Gradient Descent를 참조

**함수의 Gloabal Minimum에 도달하기 위해서는 시작점이 중요하다.** 초깃값을 어떻게 설정하는지에 따라 신경망의 학습 성능이 크게 달라질 수 있다. 이번 장에서는 학습을 시작하기 전에, 가중치의 초깃값을 어떻게 잡는것이 학습에 유리한지에 대해 알아보자.

우선 가중치를 평균이 0, 표준편차가 1인 정규분포로 가중치를 기화하는 상황을 가정해보자. 활성화 함수로 sigmoid 함수를 사용하는 5층 신경망에 무작위로 생성한 입력 데이터를 넣고 얻은 활성화 결과를 확인해보자. 각 층의 활성화 분포는 아래와 같다.

![fig10](/assets/img/weight_init/fig10.png){: w="400", h="300"}

각 층의 활성화 값이 0과 1에 치우쳐서 분포하게 된다. 하지만 Sigmoid 활성화 함수의 개형을 생각해보면 그 출력이 0또는 1에 가까워지면 미분값은 0에 가까워진다. 

![fig12](/assets/img/weight_init/fig12.png){: w="400", h="300"}

따라서 데이터가 0과 1에 치우쳐서 분포하게 되면 역전파의 기울기 값이 점점 작아지다가 사라지고 이를 **Vanishing Gradient , 기울기 소실**문제라고 한다. 특히 층이 깊어질수록 기울기 소실은 큰 문제가 된다. 

![fig13](/assets/img/weight_init/fig13.png){: w="600", h="400"}


다음은 반대로 평균이 0, 표준편차가 0.01인 정규 분포로 가중치를 초기화하는 상황을 가정하자. 각 층의 활성화 분포는 아래와 같다. 

![fig11](/assets/img/weight_init/fig11.png){: w="400", h="300"}

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

![lecun](/assets/img/weight_init/lecun.png){: w="400", h="300"}

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

![he](/assets/img/weight_init/he.png){: w="400", h="300"}

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

![xavier](/assets/img/weight_init/xavier.png){: w="400", h="300"}

> $N_\text{in}$과 $N_\text{out}$ 은 각 레이어의 입력과 출력 노드의 수를 의미한다. $N_\text{in}$이 크다면 그만큼 많은 입력값이 Weight와 곱해지고 더해지기 때문에 Activation에 들어오는 값들의 분산값이 커진다. 하지만 분산이 지나치게 큰 경우, 특히 Sigmoid와 같은 함수를 사용한다면 Gradient 가 0에 가까워져 Gradient Vanishing 문제가 발생할 수 있다.
{: .prompt-info}

이러한 문제를 해결하기 위해 3가지 방법 모두 $N_\text{in}$이 클수록 분산을 작게 하여 Weight를 0에 더 가깝게 초기화한다. 이는 순전파 과정에서 분산이 층을 지나며 급격히 커지는 문제를 방지한다. 다만 Xavier 방식에서는 $N_\text{out}$도 함께 고려하는데, 이는 역전파 과정에서 발생하는 **Gradient Exploding 문제**를 막기 위함이다. $N_\text{out}$이 크다면 $N_\text{in}$이 클때와 마찬가지로 Gradient의 분산 값이 매우 커지게 되고 이는 학습을 불안정하게 만들 수 있다. 

결론적으로 이러한 초기화 방식들은 신경망의 깊이가 깊어져도 안정적으로 학습이 진행되도록 도와준다. 적절한 초기화는 Gradient Vanishing/Exploding 문제를 해결하고 빠르고 안정적인 학습을 가능하게 한다.

[^1]: 본 포스팅은 사이토 고키의 "밑바닥부터 시작하는 딥러닝"과 수원대학교 한경훈 교수님의 강의를 참고했습니다. 
