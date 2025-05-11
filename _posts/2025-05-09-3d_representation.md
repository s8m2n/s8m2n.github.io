---
title: "3D Representation"
date: 2025-05-09
categories: [Study, Machine Learning]
tags: [3d represetation, voxel, multi-view image, point cloud, polygon mesh, implicit function, marching cube, signed distance function, poisson reconstruction]  # TAG names should always be lowercase
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

본 포스팅은 KAIST 성민혁 교수님의 CS479, Machine Learning for 3D Data 강의 Ch2,3,4를 참고했습니다[^1].

## 3D Representations
이번 강의에서는 크게 두 가지 Technology 에 대해  학습한다. 
3D Encoders: 입력으로 3D data를 받는다.
3D Decoders: 출력으로 3D Data를 내보낸다. 

![fig1](/assets/img/3d_representation/fig1.png){: w="400", h="300"}

우선 3D 데이터를 input으로 입력하기 위해서는(인코딩) 3D data를 표현하는 방식을 먼저 알아야 한다. 예를 들어 Text 데이터는 1D Sequence로 표현할 수 있고, 2D 이미지는 2D Grid의 이차원 Data로 표현할 수 있다. 3D data를 표현하는 방식에는 여러가지가 있다. 

### 3D Grids (Voxel)
가장 간단하게 생각할 수 있는 한 가지 표현 방법은 3차원에서 Convolution을 해보는 것이다. 이를 구현한 것을 Voxel, 부피가 있는 픽셀이라고 한다.

![fig2](/assets/img/3d_representation/fig2.png){: w="500", h="500"}

하지만 이 방식은 비효율적이고 복잡하여 시간이 오래 걸리는 문제 있다. 특히 우리가 3차원 공간에서 관심 있는 것은 3차원 물체의 surface 이다. 하지만 Voxel 처럼 3D 구조의 부피를 모두 표현하는 방식은 상당히 비효율적이다. 특히 Resolution을 높이기 위해 Voxel의 개수를 늘릴수록 그 비효율성은 더 커진다. 

![fig3](/assets/img/3d_representation/fig3.png){: w="400", h="300"}

이런 한계를 극복하고자 
- Architecture using adaptive data structure: Voxel을 더 효율적으로 사용하고자 함.
- SparseConvNet: Active 한 공간에서만 Convolution을 계산함.

와 같은 여러 시도들이 있지만 여전히 복잡하고 많은 메모리를 사용하는 문제가 있다.

### Multi-view Images 
3D 구조를 포착하기 위한 특별한 architecture를 사용하는 대신 오히려 간단한 방식으로 **3D를 2D로 Render 한 후 2D 데이터를 CNN으로 처리**하는 방법이다. CNN 구조를 3차원 상으로 확장 적용하는 시도로 볼 수 있다. 

![fig4](/assets/img/3d_representation/fig4.png){: w="400", h="300"}

이 방식은 오히려 Simple 한 방법이 더 효과적일 수 있음을 결과를 통해 보여준다고 한다. 

![fig5](/assets/img/3d_representation/fig5.png)

Multiview 방식을 사용해서 전체적인 형상을 render 하지 않고 일부 segment들만을 render 하고 CNN을 통해 학습하게 되면 3D 구조의 Segmentation도 가능하다고 한다.

[Kalogerakis et al., 3D Shape Segmentation with Projective Convolutional Networks, CVPR 2017](https://arxiv.org/abs/1612.02808)

특히 색, 텍스쳐, 재료 등과 같이 시각적인 appearance 정보를 잘 표현할 수 있지만, 높은 정확도를 위해서는 많은 양의 데이터가 필요하고, 기하학적인 구조를 모두 포착하지 못할 수 있다는 단점이 있다. 

### Point Cloud 
Point Cloud는 흔하게 사용하는 3D 표현법중 하나로 3D data를 독립적인 3차원 공간상의 점으로 표현하는 방식이다. 좌표 정보에 색, normal vector와 같은 정보들이 추가될 수 있다. 

![fig6](/assets/img/3d_representation/fig6.png){: w="500", h="400"}

현대에서 사용하는 대부분의 3D scanning 장비들의 출력결과는 point cloud 방식을 사용하고 이후 추가적인 조작또한 쉽기 때문에 Pyhsical simulation 분야에서도 널리 사용되는 표현법이다. 

하지만 단순한 점으로 표현을 하는 방식이기 때문에 Surface Structure가 없고 CNN과 같은 Convolutional 네트워크에 적용이 힘들다. 따라서 이를 해결하기 위해 아래와 같은 **Point Net[^2]**이라는 새로운 아키텍쳐가 제안된다.

![fig7](/assets/img/3d_representation/fig7.png)

PointNet은 Point Cloud를 Neural Network에 적용하기 위한 연구이.\
비교적 간단하고 빠르게 적용할 수 있다는 장점이 있지만 3차원 구조의 Surface 정보는 알지 못하고 이후 추가적인 변환 기법이 필요하다고 한다. 또한 높은 정확도를 위해서는 많은 수의 Point들을 필요로 한다. 

### Polygon Mesh 
Polygon Mesh는 역 유명한 3D 표현 방법중 하나로 point들을 연결하여 Connectivity를 추가하고 Surface를 생성하기 때문에 point cloud가 표현하지 못하는 surface 정보들을 표현할 수 있다는 특징이 있다. 

![fig8](/assets/img/3d_representation/fig8.png){: w="500", h="400"}

polygon mesh는 **Vertices, Edges, Faces**로 구성 되고 모든 Face가 삼각형인 경우를 특별히 Triangle Mesh라고 한다. 

![fig9](/assets/img/3d_representation/fig9.png){: w="400", h="300"}

polygon mesh는 여러 Application에 적용가능하다. 하지만 일정하지 않은 구조로 인해 유효한 mesh를 만들기 어렵다는 한계가 있다. 따라서 Regularity를 위해 CNN의 pooling Layer와 edge Contraction 이라는 방법들을 도입하는 등 다양한 시도들이 존재한다.[^3]
유효한 Mesh를 쉽게 생성하기 위해 순차적으로 모서리와 면을 생성해 Mesh를 자동적으로 만드는 Mesh Generation이라는 모델이 개발되었지만 약간의 오차로도 매우 큰 오차를 낼 수 있다는 한계가 존재한다.

![fig10](/assets/img/3d_representation/fig10.png)

지금까지의 3D Data Representation은 일종의 Explicit 한 representation이다. 

![fig11](/assets/img/3d_representation/fig11.png)

### Implicit Representation
Explicit representation과 달리, 3차원 공간 자체를 연속적인 함수로 표현하는 Implicit Representation도 있다. 

![fig12](/assets/img/3d_representation/fig12.png){: w="500", h="400"}

Implicit Function은 좌표값을 입력으로 받고, Signed Distance 혹은 Occupancy에 대한 정보를 출력하는 함수이다. 

이처럼 3D data를 표현하는 Representation은 여러가지가 있다. Implicit 함수를 계산하면 Voxel로, Mesh를 샘플링하면 Point cloud로, Mesh를 Rendering 하면 Multi-view 로 각 표현 방식들간 전환이 가능하다.

![fig13](/assets/img/3d_representation/fig13.png){: w="500", h="400"}

**만약 Voxel에서 mesh로, Point Cloud에서 Implicit Function으로도 변환**이 가능하다면 모든 기법들을 자유롭게 변환하며 3차원 데이터를 표현할 수 있을 것이다. 

## Voxels to Mesh 
### Marching Cubes 
Voxel에서 Mesh로 변환하는 Marching Cube 알고리즘에 대해 알아보자. 이 알고리즘은 1987년 발표된 알고리즘으로 Occupancy Grid가 주어질때, 물체의 표면을 근사하는 Triangle Mesh를 만드는 것이 목표다[^4]

{% include embed/youtube.html id='M3iI2l0ltbE' %}

이해를 위해 간단한 2차원 평면을 생각해보자. 2차원 평면은 Grid로 이루어져있고 각 점들은 $f(x)$라는 함수에 의해서 "Inside" 혹은 "Outside"로 정의된다. 특정한 모양을 가진 물체가 그 평면 상에 존재한다고 할때, 물체 안에 존재하는 점들은 (-), 밖에 존재하는 점들은 (+)의 부호를 갖게된다. 

$$
f(x)<0: Inside , f(x)>0: Outside
$$

그러면 자연스럽게 이웃하는 **두 점의 부호가 바뀌는 점과 점을 연결하는 선분에는 물체의 표면과 교차하는 임의의 점이 있을 것이라고 생각할 수 있다. 그 점은 간단한 선형 보간법을 이용해서 계산할 수 있다. (물론 더 복잡한 함수를 사용할 수도 있다) 그리고 그렇게 구한 점들을 연결하면 우리는 물체의 표면에 근사하는 line을 그릴 수 있게 된다.** 

![fig14](/assets/img/3d_representation/fig14.png){: w="500", h="400"}

이때 4개의 점으로 이루어지는 각 Cell들을 개별적으로 다룸으로써 연결 가능한 모든 조합들을 간단하고 효율적으로 계산할 수 있다. 특히 각 Cell들을 병렬적으로 처리하여 GPU의 Parallelize 특징을 활용할 수 있다. 한 Cell의 각 꼭짓점들은 Inside 또는 Outside가 될 수 있으므로 아래와 같이 총 $2^4=16$가지의 경우의 수를 갖는다. 

![fig15](/assets/img/3d_representation/fig15.png){: w="300", h="300"}

>이때 Inside와 Outside를 구분짓는 "선"을 어떻게 그릴 수 있는지가 관건이다. Cell의 Rotation과 Inside/Outside 간의 Inversion을 통해 16가지 경우의 수를 같은 Intersection을 그리는 경우들끼리 그룹화할 수 있다. 한 예시로 아래의 8가지 경우들은 모두 같은 경우로 간주할 수 있다.
>
>![fig16](/assets/img/3d_representation/fig16.jpg){: w="400", h="300"}
>
{: .prompt-tip}

결국 16가지의 경우들은 아래와 같이 총 4개의 그룹들로 정리할 수 있다. 따라서 우리는 이 4가지 경우들에 대한 **Look Up Table**을 만들고 그 Table을 참고해서 가능한 모든 조합을 생각할 수 있게된다. 

![fig17](/assets/img/3d_representation/fig17.png){: w="400", h="200"}

그런데 마지막 경우는 조합에 따라 Ambiguous한 두가지 경우가 생길 수 있다. 이런 경우
1. Cell을 더 작은 박스로 Subsampling 하거나
2. 두 가지 경우 중 한가지를 확률적으로 선택해서
해결할 수 있을 것이다.

![fig18](/assets/img/3d_representation/fig18.png){: w="500", h="300"}

이처럼 각 Cell에 대한 Intersection Line들의 조합을 look up table에 저장하고 이를 활용해 Surface를 추출하는 것이 Marching Cube 알고리즘의 기본 아이디어이다. 3D 차원 상에서는 Cell이 Cube가 되고, line은 plane으로 된다는 차이만 있을뿐, 2차원에서와 같은 아이디어를 적용할 수 있다. 다만 3차원에서는 총 256가지, 그룹화하면 15가지의 Cases들이 존재할 것이다. Ambiguous 한 경우는 6가지로 역 Subsampling 혹은 하나를 고르는 방식을 적용한다. 

![fig19](/assets/img/3d_representation/fig19.png){: w="500", h="500"}

Marching Cube 알고리즘의 장점은 다양한 분야에 적용가능하고 병렬적으로 처리가능하여 매우 빠르며, 쉽고 간단하게 적용할 수 있다. 또한 Voxel Grid만 입력으로 넣어주면 되기 때문에 Parameter조정으로 인한 부담이 낮다. 

반면 Sharp한 Edge는 잘 표현하지 못하고, 특별한 Cases의 경우, Look-up-table의 크기가 커질수 있다는 단점이 있다. 또한 초기모델은 정확한 기하학적 형상을 보장하지는 않는다. 

Ambiguous Cases들로 인한 단점을 보완하고자, Cube 대신 사면체를 사용한 **Marching Tetrahedra**알고리즘이 연구되기도 했다. 이 경우 4개의 모서리를 가지므로 총 16가지의 경우의 수들이 있고 그룹화하면 3가지의 그룹으로 묶을 수 있 Ambiguous한 경우가 생기지 않 장점이 있다. Marching Tetrahedra를 확장 적용한 연구로 DMTet와 같은 논문이 발표되었다. [^5]

## Point Cloud to Implicit Function
다음은 Point Cloud의 정보에서 Implicit Function을 얻는 방법을 알아보자. Voxel-to-mesh, Point Cloud-to-Implicit Function과 같은 변환 방법들을 통해 여러 3D Data Representation을 자유롭게 변환할 수 있게 된다. 

![fig13](/assets/img/3d_representation/fig13.png){: w="500", h="400"}

### Surface Normal Estimation
이번에는 Point Cloud에서 Implicit Function을 얻는 방법을 알아보자. 그러기 위해서는 Surface Normal에 대한 정보가 우선적으로 필요하다. Surface Normals은 Recovering Surface와 Volume Data에 필수적인 정보이다. 

**Tangent Plane**이란 간단히 말해서 3차원 물체 상의 임의의 점에 접하는 평면이다. 더 Formal한 정의는 아래와 같다.

| Given a point 𝐩 = 𝑥, 𝑦, 𝑧 on a surface 𝑆, for all curves passing 𝐩 and lying entirely in 𝑆,
| If the tangent lines to all such curves at 𝐩 lie on the same plane, this plane is called the tangent plane.

그리고 **Surface Normal은 Tangent Plane에 수직인 Unit Vector를 의미한다.** Tangent Plane 상의 점 𝐩 를 지나는 임의의 두 독립적인 Vector의 외적을 통해 구할 수 있다. 

![fig20](/assets/img/3d_representation/fig20.png){: w="400", h="400"}

하지만 Surface가 없고 Point 만 있는 Point Cloud에서 Surface Normal 정보는 어떻게 얻을 수 있을까?? 간단한 방식으로, Point cloud상의 임의의 점 P에 대해서 Local neighborhood와 유사 Tangent Plane을 Approximate 할 수 있다. 다시 말해, Point 주변의 Local한 Flat Plane을 가정한다. 이때 여러 Point들의 Surface Normal Vector의 정렬을 맞추기 위한 추가작업이 필요하다. 

![fig21](/assets/img/3d_representation/fig21.png){: w="500", h="300"}

하나의 Point를 가지고 가정하는 "Local"한 Plane의 범위가 어디까지인지, Point 들에 어떤 가중치를 부여하는지에 따라 결과는 상이할 수 있다. 따라서 아래와 같이 Flat한 Plane이 아닌 다변수 함수로 평면을 정의하고 Neural Network를 도입하여 Point-Wise한 weight를 결정하여 훨씬 좋은 성능을 내는 Jet Fitting과 같은 방식이 연구되었다. 

![fig22](/assets/img/3d_representation/fig22.png)

### Signed Distance Function(SDF)
3차원 공간의 Implicit 한 표현 방식의 한가지로 Occupancy Function이 있다. 안쪽이면 0, 바깥쪽이면 1로 정의하여 마치 Binary Classification 처럼 여겨질 수 있다. **Signed Distance Function은** Occupancy Function과 달리 이진 분류가 아닌 **부호를 가지는 거리로 3차원 공간을 표현하는 방식이다.**

3D Volume $ \Omega $와, Boundary Surface $\partial  \Omega $가 주어질 때, Signed Distance Function은 다음과 같이 정의할 수 있다.

![fig23](/assets/img/3d_representation/fig23.png){: w="500", h="300"}

즉 임의의 점 $\mathbf{x}$에 대해 점과 물체 표면사이의 거리를 나타내는 함수이다. 이때 점이 물체 밖에 있으면 (+)부호를, 물체 안에 있으면 (-)부호를 갖는다. 물체 표면 상에 있으면 그 점은 Surface로 $SDF = 0$이 되는 Point 일 것이다. 

Point Cloud에서 SDF를 얻는 방식은 여러가지가 있다. 각 방식들을 간략하게 알아보자.
* Simple Method: **Distance Between Point and tangent Plane**
* Local Kernel Regression: **Moving Least Squares**, Radial Basis Function etc.
* Global Reconstruction: **Poisson Surface Reconstruction**

### Simple Method
가장 간단한 방법으로는, 공간상의 임의의 점 $\mathbf{x}$와 가장 가까운 Point Cloud 상의 점 $\mathbf{p}$에 대해 점 $\mathbf{p}$로 만들수 있는 Tangent Plane과 $\mathbf{x}$의 Signed Distance 를 구하는 방법이다. 

![fig24](/assets/img/3d_representation/fig24.png){: w="500", h="300"}

하지만 이 방법은 점 $\mathbf{x}$와 가까운 하나의 점만을 사용하기 때문에 정확도가 낮을 수 있다. 

### Moving Least Squares
Simple Method의 한계를 극복하고자, **Moving Least Sqaures**방법은 Local Kernel Regression에 기반하여 여러 점들의 Wieghted Sum을 사용하는 방법이다.

{% include embed/youtube.html id='S0ptaAXNxBU' %}

즉 물체의 표면에 해당하는 샘플 Point $\mathbf{p_i}, where f(\mathbf{p_i})=0$들이 주어질 때, $f(x)$에 가장 근사하는 Implicit Function을 찾는 것이 목표이다. 

점 $\mathbf{p_i}$와 가까운 공간 상의 점 $\mathbf{x}$를 이용하여 Tayler Series로 전개하면 아래와 같다. 우리가 구하고자 하는 것은 아래 식을 만족하는 $f(x)$를 찾는 것이다. 

![fig25](/assets/img/3d_representation/fig25.png){: w="500", h="300"}

또한, 점 $\mathbf{p_i}$에서의 Gradient는 점 $\mathbf{p_i}$에서의 Surface Normal과 같다. 따라서 우리가 구하고자 하는 근사 함수 $f(x)$는 아래 식을 통해 정리할 수 있다. 즉 우리의 목표는 **Weighted Least Square Problem을 푸는 것**이다. 

![fig26](/assets/img/3d_representation/fig26.jpg){: w="400", h="300"}

이때 Weight는 다양한 방법으로 정할 수 있는데, 아래와 같은 방식으로 정의한다면, Point 와 Surface 간 거리가 클수록 더 작은 가중치 값을 갖게 된다. 

![fig27](/assets/img/3d_representation/fig27.png){: w="500", h="400"}

Moving Least Square 알고리즘은 빠르고 간단하게 적용할수 있지만 Local region에 대해 근사한 값만을 사용하고,  Weight Function에 민감하다는 단점이 있다.

### Poisson Surface Reconstruction
앞서 점 $\mathbf{p_i}$에서 Implicit Function의 Gradient는 그 점에서의 Surface Normal 과 같다는 것을 확인했다. 

![fig28](/assets/img/3d_representation/fig28.png){: w="500", h="400"}

$$
\bigtriangledown f(\mathbf{p_i})=\mathbf{n_i}
$$

그렇다면 Surface Normal 의 정보를 가지고 원함수를 복원할 수 있지 않을까 하는 것이 Poisson Reconstruction의 아이디어이다. 즉, 아래 식처럼 $\bigtriangledown f$을 적분하여 $f$를 구하는 것이 목표이다. 

![fig29](/assets/img/3d_representation/fig29.png){: w="500", h="400"}

하지만 이 방식으로 원함수를 복원하기 위해서 가능한 모든 $\bigtriangledown f=\mathbf{V}$가 적분가능해야 한다. 즉 Vector Field $\mathbf{V}$는 아래 조건들을 모두 만족하는 Conservative 한 특징을 가져야하지만 현실적으로 그러기란 쉽지 않다.[^6]

| # | Conservative Vector Field 필요충분조건                    | 수식·표현                                                                                                   |
| - | ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1 | 어떤 함수 $f$ 의 **그래디언트**일 때 | $\displaystyle \mathbf{V} = \nabla f$                                                                   |
| 2 | **컬(curl)** 이 0일 때           | $\displaystyle \nabla \times \mathbf{V} = \mathbf{0}$                                                   |
| 3 | **선적분이 경로와 무관**할 때           | $\displaystyle \int_{P_1} \mathbf{V}\!\cdot d\mathbf{r} \;=\; \int_{P_2} \mathbf{V}\!\cdot d\mathbf{r}$ |

따라서 우리는 주어진 Input에 대해서 $\bigtriangledown f(x)$와의 차이가 가장 작도록, $\hat{\mathbf{f}}$를 최소화하는 Mean Squared Error 문제를 푸는 것으로 바꿔 생각하도록 하자.

![fig30](/assets/img/3d_representation/fig30.png){: w="500", h="400"}

1차원의 경우 아래와 같은 함수를 최소화하는 문제로 생각할 수 있다.

![fig32](/assets/img/3d_representation/fig32.png){: w="500", h="400"}

#### Euler Lagrange Formulation and Poisson Equation
Euler Lagrange Equation은 간단히 말해, $\int_{\Omega }^{}L(x,f(x), f'(x))dx$ form의 Stationary Point(Min, Max, etc.)들은 $\frac{\partial L}{\partial f}-\frac{d}{dx}\frac{\partial L}{\partial f'}=0$ 과 같은 PDE의 해로 구할 수 있다는 정리이다[^7]. 

{% include embed/youtube.html id='OcRB6omfy9c' %}

1차원에서는 $L=(f'(x)-g(x))^2$이므로 아래와 같이 정리할 수 있다. 즉 Minimizing Mean Suared Error 문제는 $g$에 대한 정보가 주어졌을 때, $f"$가 $g'$와 같아지도록 하는 점들을 찾는 문제로 바꿀 수 있다. 

![fig31](/assets/img/3d_representation/fig31.png){: w="500", h="400"}

더 높은 차원에서는 이 식을 아래와 같이 바꿀 수 있고, 이 식을 **Poisson Equation**이라고 부른다. $\bigtriangledown V$는 $V$의 Divergence이고, $\bigtriangleup f$는 함수 $f$의 Laplacian으로 두 번 Divergence를 취한 연산자이다.

![fig33](/assets/img/3d_representation/fig33.png){: w="500", h="400"}

즉 최종적으로  $\bigtriangleup f=\textit{u}$를 푸는 것이 목적이다.\
하지만 Point들은 Discrete 한 점들인데 어떻게 Gradient 를 구할 수 있을까?

각 점들이 등간격 $x_{i+1}-x_{i}=h$로 배치되어 있다고 가정한다면 함수 $f$는 마치 Vector 처럼 표현할 수 있다, $\mathbf{f}=[f_1, f_2, f_3, ...,f_n]^T$ . 따라서 $x_i$에서의 Gradient는 다음과 같이 근사할 수 있다. 

![fig34](/assets/img/3d_representation/fig34.png){: w="500", h="400"}

모든 미분을 Matrix 형태로 바꿔서 표현하면 $A\mathbf{f}=\mathbf{g}$이 되고 **$A$는 미분 연산자 행렬**로 표현할 수 있다. 

![fig35](/assets/img/3d_representation/fig35.png){: w="500", h="400"}

![fig36](/assets/img/3d_representation/fig36.png){: w="500", h="400"}

$x_i$에서의 Laplacian 연산도 마찬가지로 아래와 같이 근사하고, Discrete한 함수의 matrix 연산으로 표현할 수 있다. 

![fig37](/assets/img/3d_representation/fig37.png){: w="500", h="400"}

![fig38](/assets/img/3d_representation/fig38.png){: w="500", h="400"}

정리하자면 $g$가 주어질 때, $f$에 대한 Poisson Equation $\frac{d^2f}{dx^2}=\frac{dg}{dx}$을 푸는 것은 Discrete 한 경우, $\mathbf{f}$에 대한 $L\mathbf{f}=A\mathbf{g}$을 푸는 것이다. 앞서 확인했듯, $L$, $A$, $\mathbf{g}$은 주어진다. 이 방정식에 대한 solution은 unique하지 않기 때문에 $f(p_i)=0$라 추가정보를 사용하여 더 정확한 Solution을 얻을 수 있다.

![fig39](/assets/img/3d_representation/fig39.png)
_Regularization Term을 추가한 Screened Poisson Surface Reconstruction_

Poisson Reconstruction 기법을 더 발전 시켜 Spectral Method 를 이용해 Poisson Equation을 푸는 **Shape-as-Points**[^8]와 Neural Network를 이용해 Normal 정보 없이 Point Cloud에서 Implicit 표현으로 변환하는 **Points2Surf**와 같은 연구들이 계속해서 발전되고 있다.














[^1]: [CS479 Machine Learning for 3D Data](https://www.youtube.com/watch?v=933wv8i3QKU)
[^2]: [Point Net 정리 블로그](https://mr-waguwagu.tistory.com/40)
[^3]: [MeshCNN 정리 블로그](https://m.blog.naver.com/dmsquf3015/221788095718)
[^4]: [Marching Cube 알고리즘 정리 블로그](https://xoft.tistory.com/47)
[^5]: [DMTet 정리 블로그](https://velog.io/@cjkangme/TIL-DMTet-Deep-Marching-Tetrahedra)
[^6]: [Conservative Vector Field와 적분](https://gosamy.tistory.com/239)
[^7]: [오일러 라그랑주 방정식](https://wikidocs.net/164904)
[^8]: [Shape-as-Points 정리 블로그](https://velog.io/@hbcho/Shape-As-Points-SAP-%EB%A6%AC%EB%B7%B0)




