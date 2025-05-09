---
title: "3D Representation"
date: 2025-05-09
categories: [Study, Machine Learning]
tags: [3d represetation, voxel, multi-view image, point cloud, polygon mesh, implicit function]  # TAG names should always be lowercase
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

{% include embed/youtube.html id='933wv8i3QKU' %}

본 포스팅은 KAIST 성민혁 교수님의 CS479강의를 참고했습니다. 

## 3D Representations
이번 강의에서는 크게 두 가지 Technology 에 대해  학습한다. 
3D Encoders: 입력으로 3D data를 받는다.
3D Decoders: 출력으로 3D Data를 내보낸다. 

![fig1](/assets/img/3d_representation/fig1.png){: w="400", h="300"}

우선 3D 데이터를 input으로 입력하기 위해서는(인코딩) 3D data를 표현하는 방식을 먼저 알아야 한다. 예를 들어 Text 데이터는 1D Sequence로 표현할 수 있고, 2D 이미지는 2D Grid의 이차원 Data로 표현할 수 있다. 3D data를 표현하는 방식에는 여러가지가 있다. 

### 3D Grids (Voxel)
가장 간단하게 생각할 수 있는 한 가지 표현 방법은 3차원에서 Convolution을 해보는 것이다. 이를 구현한 것을 Voxel, 부피가 있는 픽셀이라고 할 수 있다.

![fig2](/assets/img/3d_representation/fig2.png){: w="400", h="300"}

하지만 이 방식은 비효율적이고 복잡하여 시간이 오래 걸리는 단점이 있다. 특히 우리가 3차원 공간에서 관심 있는 것은 3차원 물체의 surface 이다. **하지만 Voxel 처럼 3D 구조의 부피를 모두 표현하는 방식은 상당히 비효율적**이다. 

또한 Resolution을 높이기 위해 Voxel의 개수를 늘릴수록 그 비효율성은 훨씬 커진다. 

![fig3](/assets/img/3d_representation/fig3.png){: w="400", h="300"}

이처럼 Voxel을 이용한 3D CNN구조는 너무 많은 메모리 소비와 시간을 사용한다.  이런 한계를 극복하고자 
- Architecture using adaptive data structure: Voxel을 더 효율적으로 사용하고자 함.
- SparseConvNet: Active 한 공간에서만 Convolution을 계산함.

와 같은 여러 시도들이 있지만 여전히 복잡하고 많은 메모리를 사용하는 문제가 있다.

### Multi-view Images 
또 다른 방법으로는 3D 구조를 포착하기 위한 특별한 architecture를 사용하는 대신 오히려 간단한 방식으로 **3D를 2D로 Render 한 후 2D 데이터를 CNN으로 처리**하는 방법이다. 다시 말해 기존의 CNN 구조를 사용하는 시도로 볼 수 있다. 


![fig4](/assets/img/3d_representation/fig4.png){: w="400", h="300"}

실제로 이 방식은 오히려 Simple 한 방법이 더 효과적일 수 있음을 결과를 통해 보여준다고 한다. 

![fig5](/assets/img/3d_representation/fig5.png){: w="500", h="400"}

Multiview 방식을 사용해서 전체적인 형상을 render 하지 않고 일부 segment들만을 render 하고 CNN을 통해 학습하게 되면 3D 구조의 Segmentation도 가능하다고 한다. [Kalogerakis et al., 3D Shape Segmentation with Projective Convolutional Networks, CVPR 2017](https://arxiv.org/abs/1612.02808)

이 방식은 특히 색, 텍스쳐, 재료 등과 같이 시각적인 appearance 정보를 잘 표현할 수 있지만, 높은 정확도를 위해서는 많은 양의 데이터가 필요하거나 기하학적인 구조를 모두 포착하지 못할 수 있다는 단점이 있다. 

### Point Cloud 
다른 방식으로는 3D data를 독립적인 3차원 공간상의 점으로 표현하는 방식이다. 여기서 좌표 정보에 색, normal vector와 같은 정보들이 추가될 수 있다. 

![fig6](/assets/img/3d_representation/fig6.png){: w="500", h="400"}

현대에서 사용하는 대부분의 3D scanning 장비들의 출력결과는 point cloud 방식을 사용하고 이후 추가적인 조작또한 쉽기 때문에 Pyhsical simulation 분야에서도 널리 사용되는 표현법이다. 

하지만 이 방식은 단순한 점으로 표현을 하는 방식으로 Grid Sturcture가 없고 CNN과 같은 Convolutional 네트워크에 적용이 힘들다. 따라서 이를 해결하기 위해 아래와 같은 **Point Net[^1]**이라는 새로운 아키텍쳐가 제안된다.

![fig7](/assets/img/3d_representation/fig7.png){: w="500", h="400"}

PointNet은 Point Cloud를 Neural Network에 적용하기 위한 연구로 가중 유명한 구조 중 하나이다. 비교적 간단하고 빠르게 적용할 수 있다는 장점이 있지만 이 방식 역시 3차원 구조의 Surface 정보는 알지 못하고 이후 추가적인 변환 기법이 필요하다고 한다. 또한 높은 정확도를 위해서는 많은 수의 Point 들을 필요로 한다. 

### Polygon Mesh 
Polygon Mesh는 Graphics에서 가장 유명한 3D 표현 방법중 하나로 point들을 연결하여 Connectivity를 추가하고 Surface를 생성하기 때문에 point cloud가 표현하지 못하는 surface 정보들을 표현할 수 있다는 특징이 있다. 

![fig8](/assets/img/3d_representation/fig8.png){: w="500", h="400"}

polygon mesh는 **Vertices, Edges, Faces**로 구성 되고 모든 Face가 삼각형인 경우를 특별히 Triangle Mesh라고 한다. 

![fig9](/assets/img/3d_representation/fig9.png){: w="400", h="300"}

polygon mesh는 여러 Application에 적용가능하다. 하지만 일정하지 않은 구조로 인해 Convolutional network에는 적용할 수 없다는 한계가 있다. 따라서 Regularity를 위해 CNN의 pooling Layer와 edge Contraction 이라는 방법들을 도입하는 등 다양한 시도들이 존재한다.[^2]
또한 유효한 Mesh를 만들기 어렵다는 한계도 있다. 순차적으로 모서리와 면을 생성해 Mesh를 자동적으로 만드는 Mesh Generation이라는 모델이 개발되었지만 약간의 오차로도 매우 큰 오차를 낼 수 있다는 단점이 있다.

![fig10](/assets/img/3d_representation/fig10.png){: w="500", h="400"}

지금까지의 3D Data Representation은 일종의 Explicit 한 representation이다. 

![fig11](/assets/img/3d_representation/fig11.png){: w="500", h="400"}

### Implicit Representation
Explicit representation과 달리, 3차원 공간 자체를 연속적인 함수로 표현하는 Implicit Representation도 있다. 

![fig12](/assets/img/3d_representation/fig12.png){: w="500", h="400"}

이처럼 3D data를 표현하는 Representation은 여러가지가 있다. Implicit 함수를 계산하면 Voxel로, Mesh를 샘플링하면 Point cloud로, Mesh를 Rendering 하면 Multi-view 로 각 표현 방식들간 전환이 가능하다.

![fig13](/assets/img/3d_representation/fig13.png){: w="500", h="400"}

**만약 Voxel에서 mesh로, Point Cloud에서 Implicit Function으로도 변환**이 가능하다면 모든 기법들을 자유롭게 변환하며 3차원 데이터를 표현할 수 있을 것이다. 

[^1]: [Point Net 정리 블로그](https://mr-waguwagu.tistory.com/40)
[^2]: [MeshCNN 정리 블로그](https://m.blog.naver.com/dmsquf3015/221788095718)
