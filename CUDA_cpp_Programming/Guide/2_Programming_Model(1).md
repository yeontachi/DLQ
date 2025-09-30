# Programming_Model(1)

---

# 1. Kernels
CUDA C++은 표준 C++를 확장하여, 프로그래머가 **커널(kernel)**이라는 특별한 함수를 정의할 수 있게 한다.

일반적인  C++ 함수는 호출될 때 한 번만 실행되지만, **커널 함수는 호출될 때 수많은 CUDA 스레드에 의해 동시에 실행**된다. 예를 들어 `N`개의 스레드를 지정하면 커널 함수는 `N`번 병렬로 수행한다.

### 커널 정의
커널은 다음과 같은 방식으로 정의한다.
 - `__global__` 키워드를 사용하여 함수를 선언한다.
 - 실행 시, 새로운 구문인 `<<< ... >>>` **실행 구성(execution configuration)**을 통해 몇 개의 스레드를 실행할지 지정한다.
 - 각 스레드는 고유한 **스레드 ID(thread ID)** 를 가지고 있으며, 이는 커널 내부에서 제공되는 **내장 변수(Built-in variable)**를 통해 접근할 수 있다.

대표적인 내장 변수로는 `threadIdx` 가 있다. 이는 현재 스레드의 인덱스를 알려주며, 보통 배열 인덱싱에 사용된다.

### 벡터 덧셈 예제
아래의 간단한 CUDA 코드 예시는 `N` 크기의 두 벡터 `A`와 `B`를 더해 결과를 벡터 `c` 에 저장하는 커널을 보여준다.
```cpp
// 커널 정의
__global__ void VecAdd(float* A, float* B, float* C){
	int i = threadIdx.x;  // 스레드 고유 ID(x축 기준)
	C[i] = A[i] + B[i];   // 각 스레드가 한 원소를 계산
}

int main(void){
	...
	// N개의 스레드를 생성하여 VecAdd 실행
	VecAdd<<<1, N>>>(A, B, C);
	...
}
```
 - `__global__ void VecAdd(...)` : `__global()___` 키워드는 이 함수가 GPU에서 실행될 커널임을 나타낸다.
 - `int i = threadIdx.x;` : `threadIdx`는 현재 스레드의 인덱스를 반환한다. 여기서는 1차원 블록에서 사용하므로 단순히 `0~N-1` 범위의 값이 된다.
 - `VecAdd<<<1, N>>>(A, B, C);` 
	 - `<<<1, N>>>` 구문은 **1개의 블록(block) 안에 N개의 스레드(thread)**를 실행하겠다는 의미이다.
	 - 즉, 블록 수 = 1, 블록당 스레드 수 = N
- 실행 결과:
	- `N` 개의 스레드가 동시에 실행되며, 각 스레드는 벡터 `A`와 `B`의 한 원소씩을 더해 벡터 `c`에 저장한다.
	- 따라서 전체 벡터 덧셈이 병렬로 수행된다.

정리하면, CUDA C++는 `__global__` 키워드와 `<<< >>>` 실행 구문을 통해 C++ 함수를 GPU에서 수천 개 스레드로 동시에 실행할 수 있도록 확장했다. 스레드 마다 고유한 ID(`threadIdx`)가 부여되어, 병렬 연산을 자연스럽게 데이터에 매핑할 수 있다.

위 예제처럼 벡터 덧셈과 같은 데이터 병렬 연산은 CPU에서 순차적으로 실행할 때보다 GPU에서 훨씬 빠르게 처리할 수 있다.

# 2. Thread Hierarchy
CUDA 프로그래밍 모델의 가장 중요한 특징 중 하나는 스레드를 계층적으로 조직할 수 있다는 점이다. CUDA에서 각 스레드는 `threadIdx`라는 내장 변수를 통해 자신의 위치를 확인할 수 있는데, 이 변수는 단순한 정수가 아니라 **세 개의 성분 (x, y, z)**을 가진 벡터로 정의되어 있다. 덕분에 프로그래머는 스레드를 **1차원, 2차원, 3차원 블록**으로 손쉽게 구성할 수 있으며, 이는 데이터 구조와 매우 자연스럽게 대응된다. 예를 들어, **1차원 블록**은 **벡터 연산에 적합**하고, **2차원 블록**은 **행렬이나 이미지 처리**에, 그리고 **3차원 블록**은 **볼륨 데이터나 물리 시뮬레이션과 같은 3차원 공간 데이터를 처리**하는 데 유용하다.

스레드의 인덱스와 고유 ID는 단순한 규칙에 따라 연결된다. 1차원 블록에서는 `threadIdx.x` 값이 곧 스레드 ID가 된다. 

하지만 **2차원 이상**의 블록에서는 조금 더 복잡하다. 블록의 크기가 `(Dx, Dy)`일 때, 2차원 인덱스 `(x, y)`를 가진 스레드의 고유 ID는 `x + y * Dx`로 계산된다. **3차원 블록**에서는 `(Dx, Dy, Dz)`라는 크기 안에서 `(x, y, z)` 좌표를 가지는 스레드가 있을 때, 그 스레드의 고유 ID는 `x + y * Dx + z * Dx * Dy`와 같이 정의된다. 이렇게 선형화된 ID는 모든 스레드를 하나의 일렬 배열처럼 다루고자 할 때 유용하며, 반대로 (x, y, z) 좌표는 데이터의 실제 구조와 직관적으로 대응된다.

이 개념을 행렬 덧셈 예제로 이해하면 쉽다. 두 개의 `N x N` 행렬 A와 B가 있을 때, 이들을 더해 행렬 C를 구하는 문제를 생각해보면, 
```cpp
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]){
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}

int main(void){
	...
	// Kernel invocation with one block of N * N * 1 threads
	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
	...
}
```
CPU에서라면 중첩된 두 개의 for문을 작성하여 각 행과 열을 순차적으로 계산해야 한다. 
반면 CUDA에서는 하나의 블록을 `N x N` 크기로 정의하고, 각 스레드가 고유한 `(i, j)` 좌표를 담당하도록 배치한다. 이렇게 하면 블록 안의 모든 스레드가 동시에 실행되어, 전체 행렬 덧셈이 순식간에 이루어진다. 프로그래머는 단순히 `C[i][j] = A[i][j] + B[i][j]`라는 연산을 정의하기만 하면 되고, 병렬 처리는 하드웨어가 알아서 맡는다.
### 스레드 블록과 그리드 구조
#### 블록당 스레드 개수의 한계
GPU에서 하나의 **스레드 블록(thread block)**은 동일한 **Streaming Multiprocessor(SM)** 위에서 실행된다.

따라서 블록 내부의 모든 스레드는 해당 SM의 **한정된 자원(레지스터, 공유 메모리 등)**을 공유해야 한다. 이 때문에 블록에 포함될 수 있는 스레드의 개수에는 상한이 있다. 현재의 GPU 아키텍처에서는 블록당 최대 **1024개 스레드**가 허용된다.

하지만 그렇다고 해서 프로그램 전체가 1024개의 스레드만 사용할 수 있는 것은 아니다. CUDA는 동일한 형태를 가진 여러 개의 블록을 동시에 실행할 수 있게 해 두었다. 따라서 전체 스레드 수는 **블록당 스레드 수 x 블록 수**가 되며, 이를 통해 훨씬 더 큰 데이터 집합을 다룰 수 있다.
#### 그리드의 개념과 차원 구조
블록들은 다시 모여 **그리드(Grid)**라는 더 큰 구조를 이룬다. 이 그리드는 1차원, 2차원, 또는 3차원 형태로 정의될 수 있으며, 데이터의 성격에 따라 적절히 선택된다. 예를 들어, 벡터 연산에는 1차원 그리드가 적합하고, 행렬 연산에는 2차원 그리드, 그리고 3D 볼륨 데이터 처리에는 3차원 그리드가 적합하다.

각 블록은 `blockIdx`라는 내장 변수를 통해 자신의 고유한 위치를 알 수 있고, 블록의 크기는 `blockDim`이라는 변수를 통해 접근할 수 있다. 이런 방식으로 각 스레드는 자신이 속한 블록과 그리드에서의 위치를 계산해, 어떤 데이터 요소를 처리할지를 정한다.

![alt text](/CUDA_cpp_Programming/Images/Figure4.png)

행렬 덧셈(MatAdd)을 예로 들어보자. 
이전에는 하나의 블록에 모든 스레드를 넣어 행렬의 모든 원소를 처리했지만, 행렬의 크기가 커질 경우 여러 블록으로 나누는 것이 훨신 효율적이다.

예를 들어, 블록 크기를 16 x 16으로 설정하면 하나의 블록은 256개의 스레드를 가지게 된다. 그리고 전체 `N x N` 행렬을 처리하기 위해 필요한 만큼의 블록이 자동으로 그리드에 배치된다.
```cpp
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(i < N && j < N){
		C[i][j] = A[i][j] + B[i][j];
	}
}

int main(void){
	// 블록과 그리드 정의
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	
	// 커널 실행
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
	}
```
 - `dim3 threadsPerBlock(16, 16);` : 블록당 256개 스레드(16x16)
 - `dim3 numBlocks(N/16, N/16);` : 전체 행렬을 커버하기 위해 필요한 블록 개수
 - 각 스레드는 자신이 맡은 `(i, j)` 좌표를 계산하여 `C[i][j]`에 결과를 저장한다.

각 스레드는 자신의 좌표를
`i = blockIdx.x * blockDim.x + threadIdx.x;`
`j = blockIdx.y * blockDim.y + threadIdx.y;`
와 같이 계산한다. 이를 통해 모든 스레드는 자신이 담당해야 할 행렬 원소 `(i, j)`를 정확히 찾아가며, 해당 원소의 덧셈 연산을 수행한다.
### 독립적인 블록 실행과 자동 확장성
CUDA에서 블록은 반드시 **독립적**으로 실행될 수 있어야 한다. 즉, 어떤 블록이 먼저 실행되든, 동시에 실행되든, 심지어 순차적으로 실행되더라도 프로그램의 결과는 동일해야 한다. 

이러한 독립성 덕분에 하드웨어는 블록을 어떤 순서로든, 어떤 멀티 프로세서에서든 자유롭게 스케줄링할 수 있다. 결과적으로 CUDA 프로그램은 GPU의 SM 개수와 무관하게 항상 실행 가능하며, 더 많은 SM을 가진 GPU에서는 같은 프로그램이 더 빠르게 동작하게 된다. 이것이 CUDA 모델이 제공하는 **자동 확장성(Automatic Scalability)**의 핵심이다.

### 블록 내부 협력: Shared Memory와 동기화
블록 간에는 독립성이 요구되지만, 블록 내부의 스레드들은 협력할 수 있다. 이를 가능하게 하는 도구가 바로 **공유 메모리(Shared Memory)**와 **동기화(synchronization)**이다. 공유 메모리는 각 SM에 위치한 매우 빠른 메모리 공간으로, 여러 스레드가 같은 데이터를 반복적으로 읽고 쓸 때 큰 성능 향상을 제공한다.

스레드들 또한 `__syncthreads()` 함수를 사용하여 실행을 동기화할 수 있다. 이 함수는 일종의 장벽(barrier)으로, 블록 내 모든 스레드가 해당 지점에 도달할 때까지 대기하게 만든다. 이를 통해 데이터 접근의 순서를 맞추고, 메모리 충돌을 방지할 수 있다.

CUDA는 여기서 더 나아가 **Cooperative Groups API**라는 보다 정교한 동기화 도구를 제공한다. 이를 활용하면 특정 스레드 그룹 단위의 협력까지 가능해져, 다양한 패턴의 병렬 알고리즘보다 효율적으로 구현할 수 있다.

정리하면, CUDA의 스레드 블록과 그리드 구조는 데이터 병렬성을 직관적으로 모델링할 수 있는 강력한 도구이다. 블록은 SM의 자원을 공유하기 때문에 크기에 제한이 있지만,  여러 블록을 묶은 그리드를 통해 원하는 만큼의 병렬성을 확보할 수 있다. 블록은 독립적으로 실행되므로 프로그램은 GPU 자원의 크기에 맞춰 자동으로 확장되며, 블록 내부에서는 공유 메모리와 동기화를 통해 스레드 간 협력이 가능하다. 이러한 구조적 특징이 CUDA를 **대규모 병렬 계산에 최적화된 프로그래밍 모델**로 만들어 준다.

## 2-1 Thread Block Clusters
CUDA 프로그래밍 모델은 오랫동안 **스레드 $\to$ 블록 $\to$ 그리드**라는 계층 구조를 기반으로 발전해 왔다. 하지만 **NVIDIA Compute Capability 9.0**부터닌 이 계층에 새로운 선택적 수준이 추가되었다. 바로 **스레드 블록 클러스터(Thread Block Clusters)**라는 개념이다.

스레드 블록 클러스터는 여러 개의 블록으로 이루어진 집합이다. CUDA에서 하나의 블록이 항상 동일한 **Streaming Multiprocessor(SM)** 위에서 실행되듯, 클러스터에 속한 여러 블록도 동일한 **GPU Processing Cluster(GPC)** 위에서 함께 스케줄링된다. 이는 곧 클러스터 내부 블록들이 더 밀접하게 협력할 수 있도록 보장된 실행 환경을 제공한다는 의미이다.

![alt text](/CUDA_cpp_Programming/Images/Figure5.png)

### 클러스터의 구조와 크기
스레드 블록이 1차원, 2차원, 3차원 형태로 구성될 수 있는 것처럼, 클러스터 또한 동일하게 1D, 2D, 3D 형태로 정의할 수 있다. 사용자는 한 클러스터 안에 포함될 블록의 개수를 직접 지정할 수 있으며, CUDA에서 **이식성(portability)**을 보장하기 위해 한 클러스터에 최대 8개의 블록을 둘 수 있도록 표준화되어 있다.

물론 실제 하드웨어 자원에 따라 이 한계는 달라질 수 있다. 예를 들어, 물리적으로 8개의 멀티프로세서를 지원하지 못하는 작은 GPU나, **MIG(Multi-Instance GPU)** 구성이 적용된 환경에서는 클러스터 크기가 줄어든다. 반대로 더 큰 구성을 지원하는 아키텍처에서는 8개 이상의 블록을 포함하는 클러스터도 가능하다. 이러한 세부적인 하드웨어별 지원 여부는 `cudaOccupancyMaxPotentialClusterSize` API를 통해 쿼리할 수 있다.

### 호환성과 API 지원
클러스터를 활용해 커널을 실행할 때에도, 기존의 `gridDim` 변수는 여전히 스레드 블록 단위의 크기를 나타낸다. 이는 기존 코드와의 호환성을 유지하기 위함이다. 대신, 클러스터 내부에서 각 블록의 순서를 식별하려면 **Cluster Group API**를 사용해야 한다. 이 API를 통해 클러스터 내부에서 블록의 **랭크(rank)**를 확인할 수 있고, 이를 바탕으로 블록 간 협력을 보다 정밀하게 설계할 수 있다.

### 스레드 블록 클러스터 실행 방식
CUDA에서 스레드 블록 클러스터(Thread Block Cluster)를 사용하는 방법은 크게 두 가지가 있다. 하나는 **컴파일 시점에 클러스터 크기를 지정하는 방법**이고, 다른 하나는 **런타임 시점에 동적으로 지정하는 방법**이다. 두 방식 모두 클러스터를 지원하는 GPU(Compute Capavility 9.0 이상)에서 활용할 수 있으며, 클러스터 내부 블록은 항상 동일한 GPU Processing Cluster(GPC)에 함께 배치된다.

#### 1) 컴파일 시점 클러스터 크기 지정
컴파일 시점 방법에서는 `__cluster_dims_(X, Y, Z)`라는 속성을 커널 정의에 직접 부여한다. 예를 들어 `__cluster_dims__(2, 1, 1)` 을 지정하면 X차원 방향으로 2개의 블록, Y와 Z차원 방향으로 각각 1개의 블록을 가지는 클러스터가 만들어진다.

이렇게 지정된 클러스터 크기는 **고정적**이므로, 이후 **커널 실행 시점에 변경할 수 없다**. 커널 호출 구문은 기존과 동일하게 `<<<numBlocks, threadsPerBlock>>>` 형식을 사용하며, 그리드 크기는 여전히 블록 단위로 정의된다. 다만, **그리드의 크기는 반드시 클러스터 크기의 배수**가 되어야 한다는 제약이 있다.
```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float *output)
{
	
}

int main(void){
	float *input, *output;
	// Kernel invocation with compile time cluster size
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	
	// The grid dimenstion is not affected by cluster launch, and is still enumerated
	// using number of blocks
	// The grid dimension must be a multiple of cluster size
	cluster_kernel<<<numBlocks, threadsPerBlocks>>>(input, output);
}
```
#### 2) 런타임 시점 클러스터 크기 지정
두 번째 방법은 런타임에서 `cudaLaunchKernelEx` API를 사용하는 방식이다. 이 경우 커널 정의에는 특별한 속성을 붙이지 않는다. 대신 실행 지점에서 `cudaLaunchConfig_t` 구조체를 설정하고, 그 안에 `cudaLaunchAttributeClusterDimension` 속성을 부여하여 클러스터 크기를 지정한다.

예를 들어, 속성 값으로 `x=2, y=1, z=1` 을 지정하려면, X축 방향으로 2개의 블록이 포함된 클러스터가 실행된다. 런타임 방식의 장점은 유연성이다. 동일한 커널이라도 실행 환경에 따라 다른 클러스터 크기를 적용할 수 있기 때문이다.
```cpp
// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float *output)
{

}

int main(void)
{
	float *intput, *output;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadPerBlock.x, N / threadPerBlock.y);
	
	// Kernel invocation with runtime cluster size
	{
		cudaLaunchConfig_t config = {0};
		// The grid dimenstion is not affected by cluster launch, and is still enumerated
		// using number of blocks
		// The grid dimension should be a multiple of cluster size
		config.gridDim = numBlocks;
		config.blockDim = threadsPerBlock;
		
		cudaLaunchAttribute attribute[1];
		attribute[0].id = cudaLaunchAttributeClusterDimenstion;
		attribute[0].val.clusterDim.x = 2;  // Cluster size in X-dimension
		attribute[0].val.clusterDim.y = 1;
		attribute[0].val.clusterDim.z = 1;
		config.attrs = attribute;
		config.numAttrs = 1;
		
		cudaLaunchKernelEx(&config, cluster_kernel, input, output);
	}
}
```
#### 3) 클러스터 지원 기능
Compute Capability 9.0을 지원하는 GPU에서는 클러스터 내부 블록들이 다음과 같은 추가 기능을 사용할 수 있다.
 - **하드웨어 지원 동기화** : `ClusterGroup API`를 통해 `cluster.sync()` 같은 명령으로 클러스터 내부 블록 간 동기화를 수행할 수 있다.
 - **클러스터 크기 질의** : `num_threads()`, `num_blocks()` API를 통해 클러스터 내 스레드와 블록의 수를 확인할 수 있다.
 - **스레드/블록 순서 확인** : `dim_threads()`, `dim_blocks()` API를 통해 클러스터 내 스레드와 블록의 랭크(rank)를 얻을 수 있다.
 - **분산 공유 메모리(Distributed Shared Memory)** : 클러스터에 속한 블록들은 분산 공유 메모리에 접근할 수 있으며, 이를 통해 읽기, 쓰기, 원자적 연산을 수행할 수 있다. 이 메모리는 클러스터 전체에서 활용되므로 블록 간 협력이 훨씬 강화된다.

## 2-2 Blocks as Clusters
CUDA에서 스레드 블록 클러스터를 정의하는 방법에는 여러 가지가 있다. 앞서 살펴본 `__cluster_dims__` 속성은 컴파일 시점에 클러스터 차원을 지정하는 방식이었다. 하지만 이 방식에서는 클러스터 개수가 암묵적으로 결정되어, 프로그래머가 직접 계산을 통해 전체 실행 구성을 추론해야 했다.

예를 들어 다음과 같은 코드가 있다고 하면,
```cpp
__cluster_dims__((2, 2, 2)) __global__ void foo();

// 8x8x8 클러스터, 각 클러스터는 2x2x2 스레드 블록으로 구성
foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();
```

여기서는 전체적으로 16x16x16개의 블록을 실행하는 것처럼 보이지만, 실제로는 8x8x8개의 클러스터가 실행된다. 즉, 각 차원에서 블록 수를 클러스터 크기로 나누면 전체 클러스터의 개수를 얻을 수 있는 것이다. 그러나 이 과정은 다소 번거롭고 직관적이지 않다.

### block_size 속성의 도입
이를 보완하기 위해 CUDA는 또 다른 컴파일 시점 속성인 `__block_size__`를 제공한다. 이 속성은 두 개의 튜플 인자를 받는다.
 1. **첫 번째 튜플** : 블록의 크기(threads per block)
 2. **두 번째 튜플** : 클러스터의 크기(blocks per cluster)

예를 들어 다음 코드를 보면,
```cpp
// Implementation detail of how many threads per block and blocks per cluster
// is handled as an attribute of the kernel
__block_size__((1024, 1, 1), (2,2,2)) __global void foo();

// 8x8x8 clusters.
foo<<<dim3(8, 8, 8)>>>();
```
여기서는 블록 크기와 클러스터 크기를 명시적으로 지정했기 때문에, 실행 시 `<<<dim3(8, 8, 8)>>>` 구문은 더 이상 블록 수가 아니라 **클러스터 수**를 나타내게 된다. 즉, 프로그래머는 직접 블록 수를 계산할 필요 없이 클러스터 단위로 실행을 정의할 수 있다.

### 규칙과 제약 조건
 - `__block_size__` 의 두 번째 튜플은 기본적으로 `(1, 1, 1)`로 가정된다. 따라서 필요할 때만 클러스터 크기를 지정하면 된다.
 - 스트림(stream)을 지정하려면 `<<<>>>` 구문에서 두 번째와 세 번째 인자로 각각 1과 0을 넣고 마지막 인자로 스트림을 전달해야 한다. 다른 값을 넣으면 동작이 정의되지 않는다.
 - 중요한 점은, `__block_size__`와 `__cluster_dim__`를 동시에 지정하는 것은 허용되지 않는다. 만약 `__block_size__`에 두 번째 튜플(클러스터 크기)이 지정되면, 이는 곧 **Blocks as Clusters** 모드를 활성화하는 것이며, 이때 `<<<>>>` 구문 안의 첫 번째 인자는 블록 수가 아닌 **클러스터 수**로 해석된다.