# https://github.com/kvablack/ddpo-pytorch

# ddpo-pytorch

이 프로젝트는 파이토치로 구현된 [노이즈 제거 확산 정책 최적화(Denoising Diffusion Policy Optimization; DDPO)](https://rl-diffusion.github.io/)입니다. 저순위 적응(Low-Rank Adaptation; LoRA)을 지원합니다. 원래 연구 코드는 [여기](https://github.com/jannerm/ddpo)에서 찾을 수 있지만, 이 코드는 GPU에서 실행되며 저순위 적응이 활성화된 경우, 스테이블 디퓨전(Stable Diffusion)을 미세 조정하는 데 10GB 미만의 GPU 메모리만 필요합니다!

![DDPO](teaser.jpg)

## 설치

파이썬 3.10이나 그 이상의 버전이 필요합니다.

```bash
git clone git@github.com:kvablack/ddpo-pytorch.git
cd ddpo-pytorch
pip install -e .
```

## 사용법

```bash
accelerate launch scripts/train.py
```

이 명령어를 실행하면 `ddpo_config/base.py`의 설정으로 모든 사용 가능한 GPU에서 스테이블 디퓨전 v1.5의 압축을 위해 미세 조정이 바로 시작됩니다. 각 GPU가 최소 10GB의 메모리를 가지고 있다면 작동할 것입니다. wandb에 로그인하고 싶지 않다면, 위 명령어를 실행하기 전에 `wandb disabled`를 실행할 수 있습니다.

`ddpo_config/base.py`의 기본 하이퍼파라미터는 좋은 성능을 얻기 위한 것이 아니라, 가능한 한 빨리 코드를 실행하기 위한 것입니다. 에포크(epoch)와 그래디언트 누적 단계당 훨씬 더 많은 샘플을 사용하는 것이 좋습니다.

## 중요한 하이퍼파라미터

모든 하이퍼파라미터에 대한 자세한 설명은 `ddpo_config/base.py`에서 확인할 수 있습니다. 여기서는 몇 가지 중요한 하이퍼파라미터를 소개합니다.

### prompt_fn 및 reward_fn

미세 조정 문제는 기본적으로 두 가지로 정의됩니다. 이미지를 생성할 프롬프트 집합과 이 이미지를 평가할 보상 함수입니다. 프롬프트는 인수가 없는 `prompt_fn`으로 정의되며, 매번 호출될 때마다 무작위 프롬프트를 생성합니다. 보상 함수는 배치의 이미지를 입력받아 해당 이미지의 보상 배치를 반환하는 `reward_fn`으로 정의됩니다. 현재 구현된 모든 프롬프트 및 보상 함수는 `ddpo_pytorch/prompts.py` 및 `ddpo_pytorch/rewards.py`에서 확인할 수 있습니다.

### 배치 크기 및 그래디언트 누적 단계

각 노이즈 제거 확산 정책 최적화 에포크는 이미지 배치를 생성하고, 보상을 계산한 후, 해당 이미지에 대해 몇 가지 학습 단계를 수행하는 과정으로 구성됩니다. 중요한 하이퍼파라미터 중 하나는 에포크당 생성되는 이미지의 수입니다. 평균 보상과 정책 그래드언트를 잘 추정하려면 충분한 이미지가 필요합니다. 또 다른 중요한 하이퍼파라미터는 에포크당 학습 단계의 수입니다.

하지만 해당 값은 명시적으로 정의되지 않고, 다른 하이퍼파라미터에 의해 암시적으로 정의됩니다. 우선, 모든 배치 크기는 **GPU당**입니다. 따라서 에포크당 총 생성된 이미지 수는 `sample.batch_size * num_gpus * sample.num_batches_per_epoch`입니다. 효과적인 총 학습 배치 크기 (다중 GPU를 활용하여 학습 및 그래디언트 누적을 포함하는 경우)는 `train.batch_size * num_gpus * train.gradient_accumulation_steps`입니다. 에포크당 학습 단계 수는 첫 번째 숫자를 두 번째 숫자로 나눈 값인 `(sample.batch_size * sample.num_batches_per_epoch) / (train.batch_size * train.gradient_accumulation_steps)`입니다.

(여기서 `train.num_inner_epochs == 1`이라고 가정합니다. 이 값이 더 높은 숫자로 설정된 경우, 학습은 새로운 이미지 배치를 생성하기 전에 동일한 이미지 배치로 여러 번 반복되며, 에포크당 학습 단계 수는 그에 따라 곱해집니다.)

각 학습 실행 시, 스크립트는 에포크당 생성된 이미지 수, 효과적인 총 학습 배치 크기 및 에포크당 학습 단계 수를 계산하여 출력합니다. 이 숫자들을 꼭 다시 확인하세요!

## 결과 재현

README 상단의 이미지는 저순위 적응을 활용하여 생성되었습니다! 그러나 A100 GPU 8장을 장착된 꽤 강력한 DGX 기계를 활용했으므로, 각 실험은 100 에포크 동안 약 4시간이 소요되었습니다. 단일 소형 GPU로 동일한 실험을 실행하려면 `sample.batch_size = train.batch_size = 1`로 설정하고 `sample.num_batches_per_epoch`와 `train.gradient_accumulation_steps`를 적절히 곱해야 합니다.

4개의 실험에 사용한 정확한 설정은 `ddpo_config/dgx.py`에서 확인할 수 있습니다.

```bash
accelerate launch scripts/train.py --config config/dgx.py:aesthetic
```

언어-이미지 시각 어시스턴트(Language-Image Visual Assistant; LLaVA) 프롬프트-이미지 정렬 실험을 실행하려면, [이 저장소](https://github.com/kvablack/LLaVA-server/)를 사용하여 언어-이미지 시각 어시스턴트 추론을 실행할 몇 개의 GPU를 할당해야 합니다.

## 보상 곡선

<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/593c9be3-e2a7-45d8-b1ae-ca4f77197c18" width="49%">
<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/d12fef0a-68b8-4cef-a9b8-cb1b6878fcec" width="49%">
<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/68c6a7ac-0c31-4de6-a7a0-1f9bb20202a4" width="49%">
<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/393a929e-36af-46f2-8022-33384bdae1c8" width="49%">

미적 실험에서 보시다시피, 충분히 오랜 시간 실행하면 결국 알고리즘의 불안정성을 경험하게 됩니다. 이는 학습률을 감소시켜 해결할 수 있을 것입니다. 그러나 불안정성 이후에 실제로 얻는 샘플은 질적으로 대부분 양호하지만, 평균의 하락은 몇 개의 낮은 점수를 받은 이상치 때문에 발생합니다. 이는 wandb에서 개별적으로 실행하면 확인할 수 있는 전체 보상 히스토그램에서도 명확히 나타납니다.

<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/eda43bef-6363-45b5-829d-466502e0a0e3" width="50%">
