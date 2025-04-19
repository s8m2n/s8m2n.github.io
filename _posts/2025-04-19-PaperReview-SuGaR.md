---
title: "[SuGaR] Surface Aligned 3D Gaussian Splatting"
excerpt: "본문의 주요 내용을 여기에 입력하세요"

categories: # 카테고리 설정
  - PaperReview
tags: # 포스트 태그
  - [3D Gaussian Splatting, Mesh Extraction]

permalink: /PaperReview/SugaR/ # 포스트 URL

toc: true # 우측에 본문 목차 네비게이션 생성
toc_sticky: true # 본문 목차 네비게이션 고정 여부

date: 2025-04-11 # 작성 날짜
last_modified_at: 2025-04-19 # 최종 수정 날짜
---


# SuGaR: Surface-Aligned Gaussian Spatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh-Rendering
[2024 CVPR] 

기존 Gaussian Splatting은 사실적인 뷰 합성에는 뛰어나지만, Gaussian들이 정렬되지 않아 mesh로의 변환이 어렵다는 한계를 갖는다. 본 논문은 3D Gaussian Splatting으로부터 빠르고 고품질의 Mesh를 추출하는 방법인 SuGaR를 제안한다.

[SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh...](https://arxiv.org/abs/2311.12775)

> SuGaR는 3D Gaussian Splatting으로부터 빠르고 정확하게 editable한 mesh를 추출하고, 이를 기반으로 고품질 렌더링과 직관적인 3D 장면 편집을 가능하게 하는 방법론을 제시한다. 
> We propose a method to allow precise and extremely fast mesh extraction from 3D Gaussian Splatting - SuGaR 2024, CVPR


## Introduction
NeRF 이후 3D 장면의 새로운 시점을 합성하는 방식으로 **3D Gaussian Splatting**이 주목받고 있다. 3DGS는 수백만 개의 3D Gaussian에 대해 위치, 회전, 색상, alpha blending 계수를 최적화하여 사실적인 이미지를 빠르게 생성할 수 있고 기존 NeRF에 비해 훈련 및 렌더링 속도가 빠르다는 장점이 있다. 하지만 3D Gaussian Splatting으로부터 **Scene의 Surface를 명확히 추출**하는 것은 여전히 어려운 문제다.
-   Gaussian들이 일반적으로 **불규칙하게 분포**되어 있으며
-   **표면에 정렬되지 않고**, **구조적인 형태를 띠지 않기 때문**이다.

또한 대부분의 3D 그래픽 툴은 **Mesh 기반**으로 작동한다. 따라서 여러 파이프라인에서는 Mesh를 이용한 3D Representation을 적용하지만 **Gaussian 기반 표현에서 Mesh를 추출하는 작업은 한계가 존재한다.**


> DreamGaussian_에서 적용하는 LDQ와 Marching Cube 알고리즘과 차이점?? 
   
DreamGaussian의 LDQ 방식은 3D 공간을 균등한 voxel grid로 분할하고, 각 cell에서 밀도를 계산해 Pruning한 뒤 Marching Cubes를 적용한다. 이 방식은 Gaussian이 표면에 정렬되어 있지 않기 때문에 생성된 mesh는 불규칙하고 거친 polygonal surface가 되며 부드럽고 일관된 표면을 표현하는 데 한계가 있다.

반면 SuGaR는 Gaussian을 표면에 정렬시키는 정규화 기법을 먼저 도입하고 시각적으로 의미 있는 밀도 구간만을 샘플링한 후 Poisson Reconstruction을 적용함으로써 더 정돈된 triangle mesh를 추출할 수 있다.\
또한 추출된 mesh에 Gaussian을 바인딩하고 함께 최적화함으로써 표면의 텍스처와 형상 표현을 동시에 개선한다.\
✅결과적으로 SuGaR의 mesh는 LDQ 방식보다 **구조적으로 정돈**되어 있으며(scalable and structured triangle mesh), **후속 편집 및 렌더링 응용에도 바로 활용 가능**하다.

📌**SuGaR는 Gaussian이 표면에 정렬되도록 유도하는 Regularization Term을 도입하고, 이후 Mesh를 효율적으로 추출 및 최적화하는 파이프라인을 제안**한다.

[Step 1] Surface-Aligned Regularization
  
  → 납작하고 표면에 밀착된 이상적인 Gaussian을 가정하고,
  
  → 실제 Gaussian으로부터 계산된 밀도 함수가 이 이상적인 밀도 함수와 유사해지도록 정규화 항을 설계한다.

[Step 2] Efficient Mesh Extraction

  → Gaussian의 Depth Map을 활용하여 시각적으로 보이는 영역의 밀도 level set을 효율적으로 샘플링하고,
  
  → 이를 기반으로 Poisson Reconstruction을 적용하여 고품질 Mesh를 수 분 내에 생성한다.

[Step 3] Joint Refinement with Bound Gaussians
  
  → 추출된 Mesh에 Gaussian을 바인딩하고,
  
  → Gaussian과 Mesh를 함께 최적화(jointly optimize)하여 렌더링 품질을 높이고, 직관적인 3D 편집을 가능하게 한다.

---
---

## Related Works
1. Traditional Mesh-based IBR Methods
2. Volumetric IBR Methods
3. Hybrid IBR Methods
4. Point-based IBR Methods

**3D Gaussian Splatting**
**🧠 핵심 개념**

-   장면을 **수천~수백만 개의 3D Gaussian**으로 표현
-   각 Gaussian은 위치, 색상, 스케일, 회전, 투명도(opacity) 등의 속성을 가짐    
-   뷰포인트마다 각 Gaussian을 2D 화면으로 투영하고, 알파 블렌딩(alpha blending)을 통해 이미지를 합성함

    ----------
   **✍️ 수식 표현**
    
   각 3D Gaussian g는 다음의 속성으로 구성됨:
    
   -   위치 $$\mu_g \in \mathbb{R}^3$$
   -   공분산 $$\Sigma_g \in \mathbb{R}^{3 \times 3}$$: 스케일과 방향성을 표현
   -   색상 $$c_g \in \mathbb{R}^3$$
   -   투명도 $$\alpha_g \in [0, 1]$$
    
   렌더링은 이 Gaussian들을 화면에 **splatting (뿌리는)** 방식으로 진행되며, 렌더링 식은 다음과 같이 주어진다:
    
   $$I(x) = \sum_g \alpha_g \cdot c_g \cdot \exp\left( -\frac{1}{2} (x - \mu_g)^T \Sigma_g^{-1} (x - \mu_g) \right)$$
    
   이때, 카메라 시점에서 가까운 Gaussian이 우선적으로 그려지고, 알파 블렌딩을 통해 여러 개가 겹쳐질 수 있다.
    
   ----------
   
   **🚀 장점**
   
   -   **고속 렌더링**: NeRF에 비해 훨씬 빠른 실시간 렌더링 가능
   -   **고품질**: 복잡한 텍스처나 얇은 구조까지 효과적으로 표현 가능
   -   **레이 마칭 불필요**: Volumetric 방식과 달리 ray marching 과정이 없어 효율적
   -   **간단한 아키텍처**: 신경망 없이도 고속 렌더링 달성 가능
   
   ----------
   
   ⚠️ **한계점**
   
   -   표면을 직접적으로 표현하지 않음 → **정렬되지 않은 Gaussian들**
   -   **명시적 Mesh 추출 불가능**
   -   표면 기반 편집, Deformation, 재조명 등에 **취약**
