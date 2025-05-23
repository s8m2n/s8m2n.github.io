---
title: "Gradient Descent"
date: 2025-04-30 
categories: [Study, Deep Learning]
tags: [gradient descent, learning rate, partial differnetiation, weight initialization]  # TAG names should always be lowercase
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

## 편미분, Partial Differentiation[^1]
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

이러한 문제를 해결하기 위해 새로운 알고리즘들이 등장한다. 각각의 구체적인 알고리즘에 대해서는 Optimization 에서 자세히 알아보자. 

[^1]: 본 포스팅은 사이토 고키의 "밑바닥부터 시작하는 딥러닝"과 수원대학교 한경훈 교수님의 강의를 참고했습니다. 
