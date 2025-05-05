---
title: "Optimization"
date: 2025-05-06
categories: [Study, Deep Learning]
tags: [gradient descent, optimization, sgd, momentum, rmsprop, adagrad, adam]  # TAG names should always be lowercase
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

## Optimization 알고리즘
신경망 학습의 목적은 Loss Function의 값을 가능한 한 낮추는 매개변수를 찾는 것이다. 다시 말해 매개변수의 최적값을 찾는 문제이고 이러한 문제를 푸는 것을 **Optimization, 최적화**라고 한다. 최적화 문제를 푸는 가장 간단한 방법은 매 Step마다 매개변수의 Gradient의 반대 방향으로 이동하는 경사하강법이 있다. 하지만 경사하강법(GD)은 Local Minimum 또는 Plateau에 안착하는 문제가 있었다. 이런 문제를 해결하기 위해 다양한 최적화 알고리즘들이 연구되었다. 이번 장에서는 각 알고리즘들에 대해 구체적으로 알아보자[^1].

![optimization](/assets/img/optimization/optimization.gif){: w="400", h="300"}

### Stochastic Gradient Descent(SGD), 확률적 경사 하강법
확률적 경사 하강법은 경사하강법의 한계를 개선한 방법이다. 경사하강법이 모든 데이터를 고려하여 Loss를 계산하는 반면, **SGD는 랜덤하게 데이터 하나를 선택하여 Loss를 계산한다.** 이처럼 무작위로 데이터를 선택하는 과정 때문에 Stochastic 이라는 이름이 붙는다. 

![fig9](/assets/img/optimization/fig9.png){: w="400", h="300"}

위 그림처럼 SGD는 GD 보다 덜 신중하게 방향을 설정하지만 그렇게 때문에 더 빠르게 최소점에 도달할 수 있다. 이때 그림처럼 지그재그 방향으로 나아가는 이유는 각 시점마다 모든 데이터가 아닌 일부 데이터만을 고려한 기울기를 계산하기 때문이다. 전체적으로 보면 오히려 돌아가는 것처럼 보여서 GD보다 느리지 않을까 싶지만 한번의 업데이트를 하는데 걸리는 시간인 GD에 비해 월등히 짧기 때문에 더 빠르게 최저점에 도달할 수 있다고 한다. 

이처럼 불규칙한 움직임을 갖는다는 특성 때문에 SGD는 종종 Local Minimum을 탈출할 기회를 얻을 수 있다. 따라서 복잡한 Loss 지형에서 더 유연한 탐색을 가능하게 한다. 

{% include embed/youtube.html id='VbYTp0CIJkY' %}

```
class SGD:

    """확률적 경사 하강법（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):  # 딕셔너리 형태로 params, grads 입력받는다. 
        for key in params.keys():
            params[key] -= self.lr * grads[key] # 역전파를 통해서 구한 gradient에 learning rate곱한값을 params에서 뺀다, 업데이트 
```

### Mini-Batch Gradient Descent, 미니배치 경사 하강법
하지만 SGD는 하나의 데이터만을 고려하기 때문에 대규모 데이터셋에서는 지나치게 편향된 기울기를 계산할 수 있다는 한계가 존재한다. 

따라서 **Mini-Batch GD에서는 하나의 데이터가 아닌 여러개의 데이터 묶음을 Loss 계산에 사용한다.**\
예를 들어 Batch Size가 N이라고 한다면 N개의 데이터를 랜덤하게 뽑아 그 평균을 Loss로 삼고 gradient를 계산한다. 특히 병렬 연산을 지원하는 GPU를 활용하면 Mini Batch GD의 효율성은 훨씬 높아진다. 

![fig11](/assets/img/optimization/fig11.png){: w="400", h="300"}

하지만 무작정 batch size를 키운다면 오히려 GD와 비슷해져 또 다시 Local Minimum에 빠지는 문제가 발생할 수 있기 때문에 학습 속도와 최적화 성능 사이의 Tradeoff를 고려해서 적절한 균형점을 찾아야한다. 

> batch size와 Learning Rate의 조절을 위한 흥미로운 연구가 있다고 한다[^2]. 일반적으로 Batch size가 증가할수록 Validation Error가 증가한다. 이러한 문제를 해결하기 위해 다음 두가지 방법을 제안한다. 
> 1. Linear Scaling Rule: batch size를 키울때 Learning Rate 도 비례해서 키운다.
> 2. Learning Rate Warmup: 학습 초기에 Learning Rate을 0에서 시작해서 점진적으로 증가한다.
{: .prompt-tip}

### Momentum
이름에서 알 수 있듯, 관성의 성질을 이용하는 방법이다. 이전의 GD, SGD, mini-batch GD 의 경우 현재 시점의 Gradient 만 고려한다. 하지만 이런 방법들은 특정한 상황, 특히 Loss Function이 타원형과 같은 형태일 때 문제가 발생할 수 있다. Gradient는 가장 가파른 방향을 향하기 때문에 항상 등고선에 수직하다. 따라서 타원형의 Loss Function의 경우 아래 그림과 같이 지그재그 경로로 수렴하게 된다. 

![fig6-3](/assets/img/optimization/fig6-3.png){: w="500", h="400"}

이처럼 진동하면서 수렴하는 방식은 상당히 비효율적이므로 우리는 이 진동의 폭을 줄이고자 한다. 따라서 이런 문제를 해결하기 위해 **Momentum은 이전 Gradient들을 누적하여 현재의 이동 방향을 결정한다.** 

{% include embed/youtube.html id='qfb2ezDWGIU' %}

Momentum의 동작원리를 알아보자.
1. 초기에는 GD와 동일한 방향으로 이동한다.
2. 현재와 이전 Gradient를 합산하여 방향을 결정한다. 예를 들어 첫 이동이 왼쪽이었다면, 두번째 이동이 오른쪽이어도 첫 이동의 영향으로 인해 오른쪽으로의 이동이 줄어든다.
3. 세번째 이동이 왼쪽이어도 직전의 오른쪽 이동 관성으로 인해 왼쪽으로의 이동이 줄어든다.
4. 이런 과정이 반복되면서 불필요한 진동을 상쇄하며 최소점을 향해 수렴한다.

종종 관성으로 인해 최저점을 지나치는 경우도 있지만 전반적으로 더 빠르고 효율적인 학습이 가능하다. 수식은 아래와 같다.

![eq1](/assets/img/optimization/eq1.png)

* $v_n$은 속도, $v_{n-1}$은 관성, $\alpha$는 관성계수를 나타낸다. 

Momentum 방식을 적용한 경로는 아래와 같다. 

![fig6-5](/assets/img/optimization/fig6-5.png){: w="500", h="400"}

SGD의 경로와 비교할 때 지그재그로 움직이는 정도가 덜한 것을 확인할 수 있다. 

```
class Momentum:

    """모멘텀 SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:  # 맨 처음에만 실행, 출발위치에서..
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]
```

### AdaGrad, Adaptive gradient
신경망을 학습할때, 어느 방향으로 경로를 갱신할 것인지도 중요하지만 **얼만큼 이동*할것인지, 학습률($\eta$) 역시 중요하다. 학습률이 너무 작으면 학습시간이 너무 길어지고, 반대로 너무 크면 발산해서 학습이 잘 이루어지지 않을 수 있다. 

학습률을 효과적으로 정하는 방법으로 **Learning Rate decay** 가 있다. 처음에는 크게 학습하다가 점점 학습률을 줄여나가는 방식이다. 이를 발전시킨 방법인 AdaGrad는 각각의 매개변수마다 그리고 매 step마다 학습률을 조정하는 방법이다. 

우선 각 변수에 대해서 다른 learning rate을 적용한다. 아래와 같이 $x, y$ 두 변수가 있는 상황을 가정하자. 

![fig1](/assets/img/optimization/fig1.png)

(1)과 같은 상황에서 목적지로 이동하기 위해서는 $y$축 방향으로 크게 이동해야 효율적으로 목적지에 가까워질 수 있다. 하지만 실제 함수는 매우 복잡하므로 목적지를 알기 어렵다. 따라서 AdaGrad에서는 아래와 같은 아이디어를 적용한다.

만약 (1)과 같은 상황이라면 $y$축에 대해 크게 학습이 이루어지므로 $y$에 대한 편미분 값이 클 것이고 학습 후에는 (2) 상황이 될것이다. 따라서 변수 $y$는 이미 최적점에 가까워졌고 상대적으로 적은 변화를 겪은 변수 $x$는 최적에 가려면 아직 멀었다고 생각할 수 있다. 

> 즉 큰 변화를 겪은 변수의 learning rate은 줄이고, 작은 변화를 겪은 변수의 learning rate은 키우는 방식으로 각 변수들에 대한 학습률을 갱신할 수 있다. 
{: .prompt-tip}

또한 매 step마다 학습이 진행되면서 점점 최적점에 가까워지므로 처음에는 큰 Learning Rate를 가지고 학습하다가 점점 learning rate을 줄여나가며 최적화한다. AdaGrad의 점화식은 아래와 같다. 

![e6-5](/assets/img/optimization/e6-5.png)

![e6-6](/assets/img/optimization/e6-6.png)

* $\mathbf{W}$은 갱신할 가중치 매개변수
* $\frac{\partial L}{\partial\mathbf{W}}$은 $\mathbf{W}$에 대한 손실 함수의 기울기
* $\eta$는 학습률을 뜻한다. 

여기서 $\mathbf{h}$라는 변수는 기존 기울기의 값을 제곱해서 더해주는 연산이다 (Hadamard Product)[^3]. 매개변수를 갱신할때마다 $\frac{1}{\sqrt{\mathbf{h}}}$을 곱해서 학습률 $\eta{\frac{1}{\sqrt{\mathbf{h}}}}$을 조정하는데 이는 매개변수의 원소 중에서 크게 갱신된, **즉 기울기값이 큰 원소의 학습률을 줄인다는 목적을 가진다.**

AdaGrad 방식을 적용한 경로는 아래와 같다. 

![fig6-6](/assets/img/optimization/fig6-6.png){: w="400", h="300"}

```
class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)  # error 방지용: 0이 되는 것 방지
```

### RMSProp(Root Mean Squared Propagation)
AdaGrad가 가지는 문제점은 학습이 무한히 진행될수록 $\mathbf{h_n}$이 너무 커져서 학습률이 0에 가까워지고 학습이 거의 이루어지지 않는 문제가 있다. 이를 보완하기 위한 RMSProp은 과거의 모든 기울기를 더하지 않고 먼 과거의 기울기는 서서히 잊고, 최근 기울기 정보를 크게 반영하는 방식이다. 

이를 위해 Decay rate $\gamma$를 사용한다. 점화식은 아래와 같다. 

![e1](/assets/img/optimization/e1.png)

Decay rate $\gamma$이 클수록 과거가 중요하고 작을수록 현재에 더 큰 가중치를 둔다. 

RMSProp은 각 파라미터에 대한 편미분 값을 제곱해서 누적하는 방식이다. 이는 각 파라미터들의 전체적인 이력변화를 반영하여 학습의 안정성을 높이고자 하는 시도이다.

```
class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99): #frogetting rate 가 클수록 이후의 기울기가 더 중요해진다.
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

### ADAM(Adaptive Momentum Estimation)
현재 가장 많이 사용되는 알고리즘으로 Momentum과 AdaGrad를 합친 알고리즘이다. 이전 gradient의 관성을 사용하고, learning rate의 경향을 반영한다.

$\beta_1$로 Momentum을 변형해서 아래와 같은 점화식을 세우고

![e2](/assets/img/optimization/e2.png)

$\beta_2$로 AdaGrad를 변형해서 아래와 같은 점화식을 세운다.

![e3](/assets/img/optimization/e3.png)

그리고 각 값을 보정한 값

![e4](/assets/img/optimization/e4.png)

을 사용해서 아래와 같은 점화식에 따라 경로를 갱신한다.

![e5](/assets/img/optimization/e5.png)

Adam 방식을 적용한 경로는 아래와 같다. 

![fig6-7](/assets/img/optimization/fig6-7.png){: w="400", h="300"}

```
class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
```
지금까지 여러 방식의 최적화 방법을 알아보았다.

![fig10](/assets/img/optimization/fig10.png)

어느 방식을 사용하냐는 풀어야 하는 문제에 따라 다르고 여러 하이퍼 파리미터를 어떻게 설정하느냐에 따라서도 결과가 달라진다. 

[^1]: 본 포스팅은 사이토 고키의 "밑바닥부터 시작하는 딥러닝"과 수원대학교 한경훈 교수님의 강의를 참고했습니다. 
[^2]: [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017](https://arxiv.org/abs/1706.02677)
[^3]: [Hadarmard Product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
