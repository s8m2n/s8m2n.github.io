---
title: "Back Propagation"
date: 2025-05-04
categories: [Study, Deep Learning]
tags: [back propagation, gradient descent, neural network, partial differnetiation, chain rule]  # TAG names should always be lowercase
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

모델 학습의 궁극적인 목적은 예측값과 정답 label 사이의 오차, Loss를 줄이는 것이다. 이를 위해선 모델의 각 파라미터가 Loss에 얼마나 영향을 주는지 알아야 하고 그 정보를 바탕으로 파라미터를 조정해야 한다. 가장 직관적인 방법은 수치적으로 미분을 근사해서 파라미터 하나하나의 영향을 계산하는 방식인데 이 방법은 모델이 복잡해질수록 계산량이 기하급수적으로 늘어나는 한계가 있다.\
그래서 등장한 것이 **오차역전파법(Backpropagation)**이다. 이 알고리즘은 연쇄법칙(Chain Rule)을 기반으로 모든 파라미터에 대한 미분값을 한 번의 역방향 계산으로 효율적으로 구해, 딥러닝의 학습을 가능하게 만든 핵심 기술이다[^1]

[^1]: 본 포스팅은 사이토 고키의 "밑바닥부터 시작하는 딥러닝"과 수원대학교 한경훈 교수님의 강의를 참고했습니다. 

## 계산그래프
오차역전파법을 시각적으로 쉽게 이해하는 방법은 계산그래프를 이용하는 것이다. 계산 그래프는 계산 과정을 그래프로 나타낸 그래프로 여러개의 node와 edge로 표현한다. 

사과 개당 가격 : 100, 사과 개수 : 2, 귤 개당 가격 : 150, 귤 개수 : 3, 소비세 : 10% 인 경우를 계산 그래프로 표현하면 아래와 같다. 

![fig5-3](/assets/img/back_propagation/fig5-3.png){: w="400", h="300"}

계산그래프는 이처럼 노드와 엣지로 그래프를 구성한 후, 왼쪽에서 오른쪽으로 계산을 진행하는 흐름으로 진행된다. 이 과정을 **순전파**라고 한다. 

### 왜 계산 그래프를 이용하는가??
오차역전파법을 위해서 왜 계산 그래프를 사용할까?? 계산 그래프의 중요한 특징은 **국소적 계산**이 가능하다는 점이다. 즉 계산 전체에서 어떤 일이 일어나든, 자신과 관련된 정보만을 얻을 수 있다는 점이다. 즉 전체 Loss에 대한 Local Gradient를 구할수 있다. 

![fig5-7](/assets/img/back_propagation/fig5-7.png){: w="400", h="300"}

다시 한번 우리의 목적을 상기해보자. 우리는 Loss Function을 최적화하기 위한 Parameter의 조합을 알아내는 것이 목적이다. 즉 각 변수들이 Loss 변화에 얼마나 큰 영향을 미치는지 알 수 있다면 변수들의 변화량을 통해 최종 Loss를 최적화 할 수 있을 것이다. 

위의 예시에서 사과 값을 $x$, 최종 지불금액을 $L$이라고 할때, 사과값의 변화량에 대한 최종 지불 금액의 변화량을 구하고 싶다. 식으로는 아래처럼 표현할 수 있다.

$$
\frac{\partial L}{\partial x}
$$

## 역전파, Back Propagation
역전파는 **연쇄법칙**에 따라 최종 신호에 각 노드의 국소적 미분을 곱하여 전체 Loss에 대한 각 변수의 미분을 효율적으로 계산할 수 있도록 하는 알고리즘이다. 

위 예시의 역전파 과정을 그림으로 표현하면 아래와 같다. 

![fig6](/assets/img/back_propagation/fig6.png){: w="400", h="300"}

역전파의 구체적인 계산과정을 하나씩 알아보자.

### 덧셈노드의 역전파 
아래의 간단한 예시를 통해 덧셈노드의 역전파를 알아보자.

![fig5-9](/assets/img/back_propagation/fig5-9.png){: w="400", h="300"}

$z=x+y$ 식에서 각 변수에 대한 미분은 아래와 같이 해석할 수 있다.

![e5-5](/assets/img/back_propagation/e5-5.png)

> 즉 덧셈노드의 역전파는 상류에서 전달된 값에 1을 곱하기만 할뿐이므로 입력값을 그대로 전달한다.
{: .prompt-tip}

```
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
```


### 곱셈노드의 역전파 
아래의 간단한 예시를 통해 곱셈노드의 역전파를 알아보자.

![fig5-12](/assets/img/back_propagation/fig5-12.png){: w="400", h="300"}

$z=xy$ 식에서 각 변수에 대한 미분은 아래와 같이 해석할 수 있다.

![e5-6](/assets/img/back_propagation/e5-6.png)

> 즉 곱셈노드의 역전파는 상류에서 전달된 값에 순전파시 전달되는 입력신호들을 서로 바꾼 값을 곱해서 전달한다.
{: .prompt-tip}

이전의 덧셈노드의 연산에서는 순전파의 입력신호가 필요하지 않았지만 곱셈노드의 연산을 위해서는 순전파의 입력신호가 필요하므로 곱셈노드를 구현할때는 순전파의 입력신호를 변수에 따로 저장해두어야한다. 

```
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy
```

위의 사과와 오렌지 예시를 직접 오차역전파법을 사용해서 구현해보자.

![fig8](/assets/img/back_propagation/fig8.png){: w="400", h="300"}

```
apple = 100
num_apple =2
orange = 150
num_orange = 3
tax = 1.1

mul_apple_lyr = MulLayer()
mul_orange_lyr = MulLayer()
add_apple_orange_lyr = AddLayer()
mul_tax_lyr = MulLayer()

# 순전파
apple_price = mul_apple_lyr.forward(apple, num_apple)
orange_price = mul_orange_lyr.forward(orange, num_orange)
apple_and_orange_price = add_apple_orange_lyr.forward(apple_price, orange_price)
final_price = mul_tax_lyr.forward(apple_and_orange_price, tax)

# 역전파
dfinal_price = 1
dapple_and_orange_price, dtax = mul_tax_lyr.backward(dfinal_price)
dapple_price, dorange_price = add_apple_orange_lyr.backward(dapple_and_orange_price)
dapple, dnum_apple = mul_apple_lyr.backward(dapple_price)
dorange, dnum_orange = mul_orange_lyr.backward(dorange_price)

print("final price:", int(final_price))
print("dApple:", dapple)
print("dApple_num:", int(dnum_apple))
print("dOrange:", dorange)
print("dOrange_num:", int(dnum_orange))
print("dTax:", dtax)
```

![fig7](/assets/img/back_propagation/fig7.png){: w="400", h="300"}


### 활성화 함수의 역전파
계산그래프를 신경망에 적용해서 Activation Function의 역전파를 구현해보자.

#### ReLU 계층

![fig3-9](/assets/img/back_propagation/fig3-9.png){: w="400", h="300"}

순전파시 입력 $x$가 0보다 크면 역전파는 상류의 값을 그대로 보내는 반면, 0보다 작으면 신호를 보내지 않는다. 

![fig5-18](/assets/img/back_propagation/fig5-18.png){: w="400", h="300"}

이를 구현하면 아래와 같다. 

```
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```

#### Sigmoid 계층

![fig3-7](/assets/img/back_propagation/fig3-7.png){: w="400", h="300"}

Sigmoid 함수는 $y=1/(1+exp^(-x))$와 같고 이를 계산 그래프로 그리면 아래와 같다. 

![fig5-19](/assets/img/back_propagation/fig5-19.png){: w="400", h="300"}

이를 바탕으로 Sigmoid 계층을 직접 구현해보자.

```
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
```

#### Affine/Softmax 계층
이와 같은 방법으로 Weight를 곱하고 Bias를 더하는 Affine 변환도 오차역전파법을 이용하여 구현할 수 있다. 직접 그 과정을 따라해보며 구현해보자. 

![fig5-27](/assets/img/back_propagation/fig5-27.png){: w="400", h="300"}

```
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
```


![fig5-29](/assets/img/back_propagation/fig5-29.png){: w="400", h="300"}

```
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
```

 
