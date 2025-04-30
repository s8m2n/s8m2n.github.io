---
title: "Neural Network"
date: 2025-04-30 
categories: [Study, DeepLearning]
tags: [deep learning, neural network, activation function, softmax function]  # TAG names should always be lowercase
description: Deep Learning from Scratch Ch3

math: true

# > tip, info, warning...
# {: .prompt-tip}
---
# Neural Networks 
신경망은 데이터로부터 가중치 매개변수의 적절한 값을 자동으로 학습하는 능력을 가진다. 구체적으로 어떤 식으로 가중치 매개변수를 학습하는지에 앞서, 신경망의 전체적인 구조와 특징을 살펴보자. 

## 신경망(Neural Network)
간단한 신경망을 그림으로 나타내면 아래와 같다. 

![fig_3-1](/assets/img/neural_network/fig3-1.png){: width="400" height="400"}

가장 왼쪽 층을 입력층(0층), 가운데 층을 은닉층(1층), 맨 오른쪽 층을 출력층(2층)이라고 한다.

단순한 Perceptron은 아래 그림처럼 입력 신호 $x_1, x_2$를 입력으로 받고 y를 출력한다. 이를 식으로 나타내면 [식3.1]과 같다.

![fig3-2](/assets/img/neural_network/fig3-2.png){: width="400" height="300"}
_fig3-2_

$$
y = \begin{cases}
0 & \text{if } b + w_1 x_1 + w_2 x_2 \leq 0 \\
1 & \text{if } b + w_1 x_1 + w_2 x_2 > 0
\end{cases}
$$
_[식3.1]_

$b$는 Bias를 나타내는 매개변수로 뉴런이 얼마나 쉽게 활성화되는지 그 기준을 설정한다. $w_1, w_2$는 각 신호의 가중치를 나타내는 매개변수로 각 신호의 영향력을 제어한다. 

그림 3-2를 Bias 매개변수를 포함해서 다시 그리면 그림 3-3과 같다.

![fig3-3](/assets/img/neural_network/fig3-3.png){: width="400" height="300"}

즉, 가중치가 $b$이고 입력이 1인 뉴런을 추가함으로 이 퍼셉트론의 동작은 $x_1, x_2, 1$이라는 3개의 신호가 뉴런에 입력되어 각각 가중치를 곱하여 다음 뉴런으로 전달 되는 것으로 이해할 수 있다. 

이제 [식3-1]을 더 간단한 표현으로 나타내보자. [식3-1]은 [식3-2]와 [식3-3]으로 표현된다.

$$
y = h(b + w_1 x_1 + w_2 x_2) 
$$ 
_[식3.2]_


$$
h(x) = \begin{cases}
0 & \text{if } x \leq 0 \\
1 & \text{if } x > 0
\end{cases}
$$
_[식3.3]_

입력 신호와 가중치를 곱하여 얻은 값은 $h(x)$라는 함수를 통과하고, $h(x)$ 함수는 입력이 0을 넘으면 1, 0을 안넘으면 0을 반환한다.

* $h(x)$를 활성화 함수라고 한다.

## Activation Function(활성화 함수)
활성화 함수 $h(x)$는 입력 신호의 총합을 출력신호로 변환하는 함수로 입력신호의 총합이 임계치를 넘어 활성화를 일으키는지를 정한다.

[식3-2]는 
* 가중치가 곱해진 입력신호의 총합을 계산하는 부분과: $a = b + w_1 x_1 + w_2 x_2$

* 활성화 함수에 입력해 총합의 활성화 여부를 나타내는 부분으로 나눌 수 있다: $y = h(a)$

이를 하나의 그림으로 표현하면 아래와 같다.

![fig3-4](/assets/img/neural_network/fig3-4.png){: width="400" height="300"}

입력신호와 가중치를 조합한 결과가 a노드가 되고, 활성화 함수 h를 통과하여 y라는 결과 노드가 반환된다. 

[식3.3]에서 사용하는 활성화 함수는 가장 간단한 Step Function을 사용한다. 

'''
def step_function(x):
    y=x>0
    return y.astype(np.int)
'''

![fig3-6](/assets/img/neural_network/fig3-6.png){: width="400" height="300"}


하지만 신경망은 step function 외에도 여러 활성화 함수를 사용하여 신경망을 구성할 수 있다. 

### Sigmoid Function
[식3.6]은 가장 흔하게 사용되는 활성화 함수인 Sigmoid 함수이다.

$$
h(x) = \frac{1}{1 + \exp(-x)}
$$
_[식3.6]_

sigmoid 함수는 입력을 넣으면 0~1에 해당하는 값을 반환하는 함수이다. 

'''
def sigmoid(x):
    return 1/(1+np.exp(-x))
'''

![fig3-7](/assets/img/neural_network/fig3-7.png){: width="400" height="300"}

Sigmoid 함수와 계단 함수의 가장 큰 차이점은 **매끄러움**이다. Sigmoid는 부드럽고 연속적인 곡선인 반면, 계단함수는 x=0에서 출력이 확 바뀐다. **Sigmoid 함수의 이런 매끄러움이 이후 신경망의 학습에서 아주 중요한 역할을 한다.**

![fig3-8](/assets/img/neural_network/fig3-8.png){: width="400" height="300"}

또한 계단함수는 0또는 1만 반환하는 반면 Sigmoid 함수는 0~1 사이의 연속적인 실수값을 반환한다. (이것도 두 함수의 연속성의 차이로 볼수 있다)

두 함수의 공통점으로는 모두 비슷한 모양을 가진다. 즉 입력이 작을때는 출력은 0에 가깝고, 입력이 클 때는 출력이 1에 가깝다. 

즉 입력이 중요하면 큰 값을 출력하고, 상대적으로 덜 중요하면 작은 값을 출력한다.

하지만 가장 중요한 공통점은 두 함수 모두 ***비선형 함수***라는 점이다. 

신경망에서는 활성화 함수로 비선형 함수를 사용해야 한다.

만약 선형함수를 사용한다고 하면 층을 아무리 깊게 쌓아도 결국 선형함수가 되기 때문에 층을 깊게 쌓는 의미가 없어진다. 즉 은닉층이 제 역할을 하지 못한다. 

### ReLU 함수
ReLU 함수는 최근에 자주 사용하는 활성화 함수이다. 이 함수는 입력이 0을 넘으면 입력값을 그대로 출력하고, 0이하이면 0을 출력하는 함수이다.

'''
def relu(x):
    return np.maximum(0, x)
'''

![fig3-9](/assets/img/neural_network/fig3-9.png){: width="400" height="300"}

수식으로는 [식3-7]과 같다.

$$
h(x) = \begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$
_[식3-7]_

## 신경망 구현하기 
이제 Numpy의 다차원 배열을 이용해서 신경망을 구현해보자. 다차원 배열을 간단히 말해 "숫자의 배열"이다. 

'''
import numpy as np

# 일차원 배열
A = np.array([1,2,3,4])

print(A)
print(np.ndim(A))
print(A.shape) #2차원 배열이상의 다차원 배열과 통일하기 위해 튜플을 반환한다.
print(A.shape[0])
'''

'''
# 2차원 배열
B = np.array([[1,2],[3,4],[5,6]])

print(B)
print(np.ndim(B))
print(B.shape)
print(B.shape[0])
'''

2차원 배열을 특히 **행렬**이라고 부르고 가로방향을 **행**, 세로 방향을 **열**이라고 한다. 

![fig3-10](/assets/img/neural_network/fig3-10.png){: width="400" height="300"}

행렬의 곱은 넘파이 함수 np.dot()으로 구현할 수 있다.

![fig3-11](/assets/img/neural_network/fig3-11.png){: width="400" height="300"}

'''
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])

print(A.shape, B.shape)

# C=A*B
C = np.dot(A,B)
print(C)
print(np.ndim(C))
print(C.shape)
'''

이때 주의할 점은 Matrix의 Shape에 주의해아한다. 즉 다차원 배열을 곱하려면 두 행렬의 대응하는 차원의 원소 수를 일치시켜야 한다.

![fig3-12](/assets/img/neural_network/fig3-12.png){: width="400" height="300"}

이제 신경망에서의 행렬곱을 구현해보자. 편향과 활성화 함수를 생략하고 가중치만 갖는 간단한 활성화 함수를 생각해보자.

![fig3-14](/assets/img/neural_network/fig3-14.png){: width="400" height="300"}

'''
X=np.array([1,2])
print(X)

W=np.array([[1,3,5],[2,4,6]])
print(W)

Y = np.dot(X,W)
print(Y)
'''

### 3층 신경망 구현하기

아래 그림과 같은 3층 신경망을 구현해보자.

![fig3-15](/assets/img/neural_network/fig3-15.png){: width="400" height="300"}

[tip] 표기법은 상황에 따라 다르게 표현될 수 있다. 이번 장에서는 다음과 같은 표기법을 사용한다.

![fig3-16](/assets/img/neural_network/fig3-16.png){: width="400" height="300"}

#### 입력층에서 1층으로 신호 전달

![fig3-17](/assets/img/neural_network/fig3-17.png){: width="400" height="300"}

1층으로 전달되는 노드의 수식은 다음과 같다.

$$
a_1^{(1)} = w_{11}^{(1)} x_1 + w_{12}^{(1)} x_2 + b_1^{(1)}
$$

$$
a_2^{(1)} = w_{21}^{(1)} x_1 + w_{22}^{(1)} x_2 + b_2^{(1)}
$$

$$
a_3^{(1)} = w_{31}^{(1)} x_1 + w_{32}^{(1)} x_2 + b_3^{(1)}
$$

이를 행렬식으로 표현하면 

$$
\mathbf{A}^{(1)} = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{B}^{(1)}
$$

이고 이때 행렬 $\mathbf{A}^{(1)}, \mathbf{X}, \mathbf{B}^{(1)}, \mathbf{W}^{(1)} $는 각각 다음과 같다. 

$$
\mathbf{A}^{(1)} = \left( a_1^{(1)} \quad a_2^{(1)} \quad a_3^{(1)} \right), \quad
\mathbf{X} = \left( x_1 \quad x_2 \right), \quad
\mathbf{B}^{(1)} = \left( b_1^{(1)} \quad b_2^{(1)} \quad b_3^{(1)} \right)
$$

$$
\mathbf{W}^{(1)} =
\begin{pmatrix}
w_{11}^{(1)} & w_{21}^{(1)} & w_{31}^{(1)} \\
w_{12}^{(1)} & w_{22}^{(1)} & w_{32}^{(1)}
\end{pmatrix}
$$

'''
X=np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 =np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1)+B1
'''

다음으로는 a노드의 값을 활성화 함수 h에 통과시켜 z를 출력한다. 

![fig3-18](/assets/img/neural_network/fig3-18.png){: width="400" height="300"}

'''
def sigmoid(x):
    return 1/(1+np.exp(-x))

Z1 = sigmoid(A1)

print(A1)
print(Z1)
'''

#### 1층에서 2층으로 신호 전달

다음은 1층에서 2층으로 전달되는 신호를 보자.

![fig3-19](/assets/img/neural_network/fig3-19.png){: width="400" height="300"}

'''
W2 = np.array([[0.1, 0.4], [0.2,0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
'''

#### 2층에서 출력층으로 신호 전달

![fig3-20](/assets/img/neural_network/fig3-20.png){: width="400" height="300"}

'''
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3

Y = identity_function(A3)
print(Y)
'''

지금까지의 구현을 정리해보자.

'''
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)
'''

## 출력층 설계하기

신경망은 분류와 회귀 모두 사용할 수 있다. 다만 어떤 문제냐에 따라서 출력층에 사용하는 활성화 함수가 달라진다. 
* 일반적으로 회귀에는 항등함수를,
* 분류에는 SoftMax 함수를 사용한다.


### SoftMax 함수 구현하기
항등함수는 입력을 그대로 출력한다. 

![fig3-21](/assets/img/neural_network/fig3-21.png){: width="400" height="300"}

반면 분류에서 사용하는 **SoftMax 함수**의 식은 다음과 같다.

$$
y_k = \frac{\exp(a_k)}{\sum_{i=1}^{n} \exp(a_i)}
$$

이를 그림으로 나타내면 아래와 같다. 

![fig3-22](/assets/img/neural_network/fig3-22.png){: width="400" height="300"}



'''
def softmax_raw(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y 
'''

>Softmax 함수를 구현할 때 주의할 점이 있다. Softmax 함수는 지수함수를 사용하는데, 종종 너무 큰 값이 반환되는 경우가 있다. 그리고 이런 큰 값을 나눠버리면 컴퓨터는 Overflow 문제를 겪고 불안정한 결과가 나올 수 있다.
{: .prompt-warning}

따라서 이러한 문제를 해결하기 위해 임의의 정수 C를 더하거나 빼서 Overflow 문제를 해결하고자 한다. C는 일반적으로 입력신호의 최댓값을 사용한다. 

'''
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a

    return y
'''

'''
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)
print(np.sum(y))
'''

softmax함수의 특징적인 점은 **출력의 총합이 1**이라는 점이다. 이 성질 덕분에 softmax 함수의 출력을 **확률**로 해석할 수 있다. 

| 즉, softmax 함수를 이용하여 특정 클래스의 점수(score)를 확률적(통계적)으로 대응시킬수 있게된다. 

> 기계학습의 문제 풀이는 **학습**과 **추론**두단계로 이루어진다. 학습 단계에서는 모델을 학습하여 가중치를 최적화하고, 추론 단계에서는 학습한 모델로 미지의 데이터에 대한 추론을 수행한다. 이때 Softmax 함수는 학습 과정에서 예측값과 레이블 값 사이의 Loss를 줄이기 위해 사용되므로 학습시에만 사용하고 추론 시에는 보통 생략하는 경우가 많다.
(: .prompt-info}






