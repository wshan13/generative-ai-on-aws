# https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/schedulers/scheduling_ddim.py에서 복사됐습니다. 다음과 같은 수정이 포함됐습니다.
# - U-Net을 활용한 예측값을 기반으로 `prev_sample`의 로그 확률을 계산해 반환합니다.
# - `variance_noise` 대신, `prev_sample`을 선택적 인수로 받습니다. `prev_sample`이 제공된 경우 이를 사용해 로그 확률을 계산합니다.
# - 타임스텝(timesteps)은 배치된 `torch.Tensor`로 처리될 수 있습니다.


from typing import Optional, Tuple, Union

import math
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def ddim_step_with_logprob(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    확률 미분 방정식(Stochastic Differential Equations; SDE)를 역으로 적용해 이전 타임스텝에서의 샘플을 예측합니다.
    이는 학습된 모델 출력(대부분 예측된 노이즈)을 기반으로 확산 과정을 진행하는 핵심 함수입니다.


    인수:
        model_output (`torch.FloatTensor`): 학습된 확산 모델 출력.
        timestep (`int`): 확산 체인의 현재 특정 타임스텝.
        sample (`torch.FloatTensor`): 확산 과저에서 생성된 현재 샘플.
        eta (`float`): 확산 단계에서 추가되는 노이즈의 가중치.
        use_clipped_model_output (`bool`): `True`인 경우, 클리핑된 예측 원본 샘플에서 "수정된" `model_output`을 계산합니다.`self.config.clip_sample`가 `True` 일때 예측된 원본 샘플이 [-1, 1]로 클리핑될 때 필요합니다.
             클리핑이 일어나지 않은 경우, "수정된" `model_output`은 입력으로 제공된 것과 일치하며, `use_clipped_model_output`은 영향을 미치지 않습니다.
        generator: 랜덤 생성기. 
        variance_noise (`torch.FloatTensor`): `generator`를 활용해 분산을 통한 노이즈를 생성하는 대신, 분산 자체에 대한 노이즈를 직접 제공할 수 있습니다.
             이는 CycleDiffusion과 같은 방법에 유용합니다. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): DDIMSchedulerOutput 클래스를 반환하는 대신 튜플을 반환하는 옵션.

    반환값:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        `return_dict`가 True일 경우, [`~schedulers.scheduling_utils.DDIMSchedulerOutput`]를 반환하고, 그렇지 않으면 `tuple`을 반환합니다.
        튜플을 반환하는 경우, 첫 번째 요소는 샘플 텐서입니다.

    """
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # DDIM 논문(https://arxiv.org/pdf/2010.02502.pdf)의 공식 (12)와 (16)을 참조하세요.
    # 이상적으로는 DDIM 논문을 자세히 읽어 이해하시길 바랍니다.

    # 기호 (<변수명> -> <논문에서 언급한 변수명>)
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. 이전 단계 값(=t-1)을 가져옵니다.
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # gather에서 OOB(Out of Bound)를 방지하기 위해 활용
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. 알파와 베타를 계산합니다.
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)

    beta_prod_t = 1 - alpha_prod_t

    # 3. 예측된 노이즈로부터 예측된 원본 샘플을 계산합니다.
    # 이는 https://arxiv.org/pdf/2010.02502.pdf 논문의 공식 (12)에서 "predicted x_0"라고도 합니다.
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. "predicted x_0"은 클립하거나 임계값으로 제한
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. 분산을 계산합니다. "sigma_t(η)" -> 공식 (16) 참고
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # Glide에서는 pred_epsilon이 항상 클립된 x_0에서 다시 유도됩니다.
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. https://arxiv.org/pdf/2010.02502.pdf 논문의 공식 (12)에서 "x_t로 향하는 방향"을 계산
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. https://arxiv.org/pdf/2010.02502.pdf 논문의 공식 (12)에서 "랜덤 노이즈 없이 x_t"를 계산합니다.
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # prev_sample_mean과 std_dev_t를 주어진 prev_sample 로그 확률을 계산.
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # 배치 차원을 제외한 모든 차원에 대해 평균을 계산합니다.
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob
