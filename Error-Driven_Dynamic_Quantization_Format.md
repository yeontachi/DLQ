@# 1. 관련 Baseline 정리 & 비교

### 1-1. 대표 Baseline들 (CNN/ViT/LLM)

```text
[QAT 계열]
- LSQ (2019): Learnable Step Size Quantization
  · scale(step size)를 학습해서 양자화 오차 최소화
  · 레이어별로 독립적인 local optimization

- HAWQ / HAWQ-V2 (2019~): Hessian 기반 layer-wise mixed-precision
  · 민감도(Hessian spectrum)로 레이어별 비트 폭 결정
  · global하게 bit allocation은 하지만, 에러 전파를 직접 쓰진 않음

[PTQ 계열 - CNN/ViT 중심]
- DFQ, ZeroQ (2019–2020): 데이터 없는 PTQ
  · activation 통계/합성 데이터로 scale 추정
- AdaRound (2020): rounding 위치를 continuous 변수로 최적화
- BRECQ (2021): block reconstruction 기반 PTQ
  · block 단위로 weight/activation 재구성
- PTQ4ViT (2022): ViT 분포 특성에 맞춘 PTQ
- OmniQuant (2023): calibrating both weights & activations with advanced calibration

[LLM / Transformer / VLM 계열]
- ZeroQuant (2022): LLM PTQ with mixed optimization techniques
- SmoothQuant (2022): activation outlier를 weight로 넘기고 α로 조정
- GPTQ (2022): weight-only PTQ, 2nd-order 정보 기반 greedy 최적화
- AWQ (2023): important weight subset 보호, activation-aware W-only PTQ
- QLoRA (2023): 4bit NF4 + LoRA 기반 효율 fine-tuning
```

### 1-2. 네 아이디어와 Baseline의 차이 포인트

축을 몇 개 잡고 비교해보면:

| 축                     | 기존 LSQ/BRECQ/OmniQuant 등                       | SmoothQuant / AWQ / GPTQ               | **너의 아이디어 (Error-driven Format PTQ)**        |
| --------------------- | ---------------------------------------------- | -------------------------------------- | -------------------------------------------- |
| 최적화 대상                | 주로 각 레이어의 scale / rounding / bitwidth          | outlier 처리가중, 중요 weight 보호             | **다음 레이어의 quantization format & zero-point** |
| 오차 활용 방식              | 레이어 내부 에러를 최소화(local reconstruction)           | 일부는 loss 기반 tuning, 일부는 heuristic      | **레이어 i의 오차를 레이어 i+1의 포맷 결정 기준으로 사용**        |
| global error 관점       | 일부 mixed-precision(HWQ)만 global bit allocation | 대부분 local block/weight 수준              | **error 흐름 자체를 global하게 제어하는 개념**            |
| 포맷 전환(symmetric↔asym) | activation 분포/범위에 따라 정적 결정                     | activation outlier 유무에 따라 정적/heuristic | **에러 크기에 따라 dynamic하게 포맷 스위칭 + Z 이동**        |
| zero-point 설계         | min/max 기반, 혹은 간단한 통계                          | 보통 standard asymmetric 방식              | **δW 평균을 반영하여 “오차 흡수용” zero-point 재배치**      |

핵심은:

> 기존 연구는 **“이 레이어 안에서 오차를 줄이자”**에 집중되어 있고,
> 네 아이디어는 **“이 레이어에서 생긴 오차를 다음 레이어 포맷 설계에 사용하자”**라는 완전히 다른 관점이라는 것.

이게 딱 “global error flow control” 느낌이라 꽤 신선하다.

---

## 2. 연구 Proposal 초안 (졸업논문/학부연구용)

### 2-1. 연구 배경 및 필요성

최근 딥러닝 모델의 규모가 급격하게 증가함에 따라, 학습과 추론 과정에서 요구되는 연산량과 메모리 사용량 또한 함께 증가하고 있다. 특히, 임베디드 디바이스나 온디바이스 환경에서는 전력, 메모리, 레이턴시 제약이 크기 때문에, 모델을 경량화하는 양자화(Quantization) 기술이 필수적인 요소로 자리 잡고 있다.

기존의 양자화 연구는 크게 QAT(Learnable Step Size, LSQ 계열)와 PTQ(BRECQ, GPTQ, SmoothQuant, AWQ 등)로 나뉘며, 각각 레이어 단위에서 스케일, 라운딩, 비트 폭을 최적화하여 성능 저하를 최소화하는 데 초점을 맞추고 있다. 하지만 대부분의 방법은 **각 레이어를 독립적인 최적화 단위**로 취급하고, 한 레이어에서 발생한 양자화 오차가 **다음 레이어로 어떻게 전파되는지에 대한 “오차 흐름(error flow)” 관점은 충분히 탐구되지 않았다.**

특히 초저비트(4bit 이하)에서는 한 레이어에서의 작은 오차가 다음 레이어에서 증폭되며 성능 저하를 유발하는데, 현재 PTQ 방법들은 이러한 오차 전파를 **“사후적으로” 줄이는 방식**에 머물러 있다. 따라서, 양자화 오차를 단순히 최소화하는 수준을 넘어, **네트워크 전반에서의 오차 흐름을 제어하는 새로운 PTQ 프레임워크**가 필요하다.

---

### 2-2. 연구 목적

본 연구의 목적은 다음과 같다.

1. 레이어 (L_i)에서 발생한 양자화 오차 (\delta W_i)를 명시적으로 추정한다.
2. 이 오차를 다음 레이어 (L_{i+1})의 **양자화 포맷(비트 폭, 대칭/비대칭, zero-point 위치)**을 결정하는 신호로 활용하여, 네트워크 전반의 오차 흐름을 제어하는 **Error-Driven Dynamic Quantization Format (EDDQF)** 기법을 제안한다.
3. 제안 기법이 CNN(ResNet), ViT, 그리고 가능하다면 소형 LLM 또는 VLM 구조에서, 기존 PTQ 기법 대비 성능 및 안정성 측면에서 유의미한 개선을 보이는지 검증한다.

---

### 2-3. 선행연구 및 한계 정리

1. **LSQ / LSQ+ (QAT)**

   * 장점: 학습 과정에서 양자화 스케일을 함께 학습함으로써 높은 정확도를 유지할 수 있다.
   * 한계: 학습 데이터와 추가 학습이 필요하며, 온디바이스/실제 배포 환경에서는 QAT 비용이 부담스럽다.

2. **BRECQ / AdaRound / OmniQuant (PTQ)**

   * 장점: 사전 학습된 모델에 대해 비교적 적은 수의 calibration 데이터로 높은 정확도의 PTQ를 달성한다.
   * 한계: 대부분 block 또는 layer 내부에서의 reconstruction에 초점을 맞추며, 레이어 간 오차 전파 구조는 고려하지 않는다.

3. **SmoothQuant / AWQ / GPTQ (LLM, Transformer 중심)**

   * SmoothQuant: activation outlier를 weight로 흡수하는 방식으로 LLM의 activation quantization을 안정화한다.
   * AWQ: 일부 중요한 weight subset을 고정 정밀도로 유지하여 전체 성능을 보호한다.
   * GPTQ: 2차 정보 기반 greedy 최적화를 통해 weight-only PTQ를 수행한다.
   * 한계: 이들 기법도 각각 local한 관점의 최적화에 해당하며, “이전 레이어의 오차를 다음 레이어 포맷 설계에 반영”하는 구조는 갖고 있지 않다.

> 요약하면, **기존 방법들은 모두 “오차의 크기를 줄이는 것”에 집중**하지만,
> **오차의 방향과 구조를 이용해서 다음 레이어를 설계하는 “오차 흐름 제어” 관점은 비어 있다.**

---

## 3. 제안 방법: Error-Driven Dynamic Quantization Format (EDDQF-PTQ)

### 3-1. 기본 아이디어

레이어 (L_i)의 weight quantization에서 발생한 오차를
[
\delta W_i = W_i - \hat{W}_i
]
라고 정의한다. 여기서 (W_i)는 FP32 weight, (\hat{W}_i)는 quantized weight이다.

이때 (\delta W_i)의 통계량(예: 평균, 평균 절댓값, 분산 등)을 계산하여, **다음 레이어 (L_{i+1})에서 사용할 양자화 포맷**을 다음과 같이 동적으로 결정한다.

* (\eta_i = \mathbb{E}[|\delta W_i|]) (또는 (\eta_i = \mathbb{E}[\delta W_i^2]))가 특정 임계값 (\tau) 이상이면:

  * (L_{i+1})의 activation quantization을 **대칭(symmetric) → 비대칭(asymmetric)**으로 강제 전환
  * zero-point (Z_{i+1})를 (\delta W_i) 평균에 가까운 위치로 이동하여 오차를 “상쇄”하도록 설계

### 3-2. Error-aware format switching

임계값 조건:

[
\eta_i = \mathbb{E}[|\delta W_i|]
]
[
\text{If } \eta_i > \tau, \quad \Rightarrow \quad \text{format}(L_{i+1}) = \text{Asymmetric}
]
[
\text{Else}, \quad \text{format}(L_{i+1}) = \text{(default, e.g. symmetric)}
]

여기서 (\tau)는

* 절대 기준값 (예: 0.01, 0.02)
* 또는 weight 스케일 대비 상대값 (예: (\eta_i / \mathbb{E}[|W_i|] > \alpha))

등으로 정의할 수 있다.

### 3-3. Zero-point relocation (오차 흡수용 Z 이동)

비대칭 양자화에서 activation (a_{i+1})에 대해,

[
\hat{a}*{i+1} = s*{i+1}(q_{i+1} - Z_{i+1}), \quad q_{i+1} \in {0, \dots, 2^{B_{i+1}} - 1}
]

이라고 할 때, 기존 방식은 대개 min/max 기반으로 Z를 잡는다.
제안 방식은 다음과 같이 설계한다:

[
\mu_{\delta,i} = \mathbb{E}[\delta W_i]
]
[
Z_{i+1} = \text{round}\left( \frac{\mu_{\delta,i}}{s_{i+1}} \right)
]

즉, 레이어 i에서 발생한 평균 오차 방향으로 zero-point를 이동시켜,
다음 레이어에서의 활성값이 이 오차를 부분적으로 상쇄하도록 만드는 효과를 기대한다.

### 3-4. 선택적 확장: Bitwidth adaptation

추가 확장으로,

* (\eta_i)가 매우 크면 (B_{i+1})을 1bit 증가 (예: 4bit → 5bit or 6bit)
* 반대로 (\eta_i)가 매우 작으면 (B_{i+1})을 유지 혹은 감소

하는 **error-aware mixed precision PTQ**로 확장할 수 있다.
이 부분은 time/resource 여유가 있으면 “추가 실험” 정도로 넣으면 매우 멋진 결과가 된다.

### 3-5. 알고리즘 개요 (의사코드)

```pseudo
for i in reversed(Layers):  # or forward, 설계에 따라
    # 1. Layer i weight quantization
    W_i_fp32 = W[i]
    W_i_q, scale_w_i = quantize_weight(W_i_fp32, format=base_format, bit=B_w)

    # 2. Quantization error 측정
    delta_W_i = W_i_fp32 - W_i_q
    eta_i = mean(abs(delta_W_i))
    mu_delta_i = mean(delta_W_i)

    # 3. 다음 레이어 activation 포맷 결정
    if i < num_layers - 1:
        j = i + 1
        if eta_i > tau:
            act_format[j] = "asymmetric"
            # 기존 act scale s_{j} 는 calibration 데이터로 추정
            s_j = act_scale[j]
            Z_j = round(mu_delta_i / s_j)
            zero_point[j] = clip_Z_into_valid_range(Z_j, B_act)
        else:
            act_format[j] = "symmetric"
            zero_point[j] = 0
```

실제로는 calibration 데이터로 activation 통계를 먼저 뽑은 뒤,
이 루프를 돌면서 format, Z를 업데이트하는 구조로 구현할 수 있다.

---

## 4. 실험 계획

### 4-1. 모델 및 데이터셋

1. **CNN (기초 실험)**

   * 모델: ResNet-18, ResNet-50
   * 데이터셋: CIFAR-10(빠른 프로토타이핑), ImageNet-1K (subset or full)

2. **Vision Transformer**

   * 모델: ViT-Tiny / ViT-Small / DeiT-Small (HuggingFace or timm)
   * 데이터셋: ImageNet (또는 ImageNet-subset)

3. **선택사항: 소형 LLM / VLM**

   * 예: 1~3B 정도의 small LLM (실제 계산 자원 따라 조정)
   * 또는 CLIP-like VLM에서 vision branch에만 적용

### 4-2. 비교 Baseline

* FP32 (upper bound)

* 기본 PTQ:

  * Symmetric-only 8bit / 4bit
  * Asymmetric-only 8bit / 4bit

* 기존 고급 PTQ:

  * BRECQ (CNN)
  * PTQ4ViT or OmniQuant 스타일 (ViT)
  * SmoothQuant / AWQ (Transformer/LLM/VLM)

* **제안 방법:**

  * EDDQF-PTQ (Error-Driven Dynamic Quantization Format)

    * variant A: format only (symmetric ↔ asymmetric)
    * variant B: format + zero-point relocation
    * (선택) variant C: + bitwidth adaptation

### 4-3. 평가 지표

* Top-1 / Top-5 Accuracy (classification)
* Per-layer / per-block reconstruction error
* Calibration data 수에 따른 성능 변화
* 비트 폭 / 메모리 사용량 / 연산량 (MAC 수, bit-ops 등)
* (선택) LLM: Perplexity, zero-shot 평가 등

### 4-4. Ablation Study 계획

1. **오차 기준의 효과**

   * no-error, 단순 분포 기반 포맷 선택 vs δW 기반 포맷 선택
2. **Z shift 유무**

   * Z = 0 (symmetric) vs min-max 기반 asymmetric vs δW 기반 asymmetric
3. **임계값 (\tau) 변화**

   * 낮은 τ / 중간 τ / 높은 τ 에 따른 결과 비교
4. **어느 레이어에 적용할 것인가**

   * 모든 레이어에 적용 vs 특정 민감한 block만 적용 (예: 첫 conv, 마지막 block 등)

---

## 5. 구현 계획 및 일정 예시

### 1단계 (1–2주): 기본 PTQ 파이프라인 구축

* PyTorch 기반:

  * ResNet-18 FP32 inference + PTQ (symmetric/asymmetric) baseline 구현
  * calibration 데이터 수십~수백 장 기반 activation 통계 수집
* δW_i 계산 코드 + per-layer error logging

### 2단계 (2–3주): EDDQF-PTQ 코어 구현

* δW_i → η_i, μ_δ,i 계산 모듈
* format switching & zero-point relocation 로직 구현
* ResNet-18 on CIFAR-10에서 빠른 예비 실험

### 3단계 (3–4주): ViT/대형 모델로 확장

* ViT-Tiny / DeiT-Small에 적용
* 4bit / 8bit 실험
* baseline(BRECQ/OmniQuant 스타일)을 최대한 간단하게 재현

### 4단계 (2–3주): Ablation & 분석

* τ 값 스윕
* format-only vs format+Z vs +bitwidth 비교
* error 흐름 시각화 (layer index vs η_i, 최종 accuracy 등 그래프)

### 5단계 (2주): 논문/보고서 작성

* 실험 결과 정리, figure/table 생성
* 논문 구조에 맞춰 Introduction ~ Conclusion 작성
* 블로그 글 모드로 재작성(후속 작업)

---

## 6. 기대 효과 및 기여

1. **새로운 PTQ 패러다임 제시**

   * 레이어별 오차를 다음 레이어 포맷 설계에 사용하는 **“오차 흐름 기반 PTQ”** 개념 최초 제안.

2. **초저비트에서의 안정성 향상**

   * 특히 4bit activation + weight 설정에서 기존 symmetric-only 보다 의미 있는 성능 개선 기대.

3. **간단하지만 강력한 heuristic**

   * 구현이 매우 단순(평균/평균절댓값만으로 결정)하면서도, 다양한 네트워크 구조(CNN, ViT, Transformer)에 적용 가능.

4. **향후 확장성**

   * RL 기반 bitwidth 결정, 하드웨어-aware 포맷 선택 등으로 자연스럽게 확장 가능한 “핵심 아이디어” 제공.

---

## 7. 논문 형식 구조 예시 (TOC)

```text
[Working Title]
"Error-Driven Dynamic Quantization Formats for Post-Training Quantization"

1. Introduction
   1.1 Background and Motivation
   1.2 Limitations of Existing PTQ Methods
   1.3 Contributions

2. Related Work
   2.1 Quantization-Aware Training
   2.2 Post-Training Quantization for CNNs and ViTs
   2.3 LLM and VLM Quantization (SmoothQuant, GPTQ, AWQ 등)
   2.4 Discussion: Gap in Global Error Flow Control

3. Proposed Method: Error-Driven Dynamic Quantization Format (EDDQF-PTQ)
   3.1 Layer-wise Quantization Error Estimation
   3.2 Error-Aware Format Switching (Symmetric ↔ Asymmetric)
   3.3 Zero-Point Relocation Using δW Statistics
   3.4 Optional: Error-guided Bitwidth Adaptation
   3.5 Algorithm and Implementation Details

4. Experimental Setup
   4.1 Models and Datasets
   4.2 Baseline Methods
   4.3 Evaluation Metrics
   4.4 Implementation Details

5. Results
   5.1 CNN (ResNet-18/50) on CIFAR-10 / ImageNet
   5.2 ViT / DeiT on ImageNet
   5.3 (Optional) Transformer/LLM/VLM Experiments
   5.4 Ablation Studies
       - Effect of δW-based format switching
       - Effect of zero-point relocation
       - Sensitivity to τ
   5.5 Discussion

6. Conclusion and Future Work
   6.1 Summary of Findings
   6.2 Limitations
   6.3 Future Directions (RL-based bitwidth, HW-aware integration 등)
```

