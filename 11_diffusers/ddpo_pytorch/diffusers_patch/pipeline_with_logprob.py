# https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py의 diffusers 저장소에서 가져온 코드입니다. 아래와 같은 수정 사항이 있습니다.
# - `ddim_with_logprob.py`에서 수정된 버전인 `ddim_step_with_logprob`를 사용합니다. 따라서 이 코드는 `ddim` 스케줄러만 지원합니다.
# - 생성 과정에서 각 노이즈가 제거된 단계의 로그 확률 뿐만 아니라 노이즈 제거 과정의 모든 중간 잠재 변수를 반환합니다.

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from .ddim_with_logprob import ddim_step_with_logprob


@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    이 함수는 파이프라인을 호출해 이미지 생성 작업을 수행할 때 호출됩니다.

    인:
        prompt (`str` or `List[str]`, *optional*):
            이미지 생성을 안내할 프롬프트 또는 프롬프트들입니다. 정의되지 않은 경우, `prompt_embeds`를 대신 전달.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            생성된 이미지의 높이(픽셀 단위).
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            생성된 이미지의 너비(픽셀 단위).
        num_inference_steps (`int`, *optional*, defaults to 50):
            노이즈 제거 단계 수입니다. 더 많은 노이즈 제거 단계는 일반적으로 더 높은 품질의 이미지를 생성하지만, 추론 속도는 느려집니다.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            [분류기를 활용하지 않은 확산 안내(Classifier-Free Diffusion Guidance)](https://arxiv.org/abs/2207.12598) 정의된 안내된 스케일입니다.
            `guidance_scale`은 [Imagen 논문](https://arxiv.org/pdf/2205.11487.pdf)에서 식 2의 `w`로 정의됩니다.
            안내된 스케일은 `guidance_scale > 1`로 설정해 활성화됩니다. 더 큰 안내된 스케일은 텍스트 프롬프트와 밀접하게 연관된 이미지를 생성하도록 유도하지만, 일반적으로 이미지 품질이 낮아질 수 있습니다.
        negative_prompt (`str` or `List[str]`, *optional*):
            이미지 생성을 지침하지 않는 프롬프트 또는 프롬프트들입니다.
            정의되지 않는 대신 `negative_prompt_embeds`를 전달해야 합니다. `guidance_scale`이 1보다 작은 경우에 무시됩니다.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            각 프롬프트당 생성될 이미지 수입니다.
        eta (`float`, *optional*, defaults to 0.0):
            DDIM 논문(https://arxiv.org/abs/2010.02502)에서의 파라미터 eta (η)에 해당합니다.
            [`schedulers.DDIMScheduler`]에만 적용되며, 다른 스케줄러에서는 무시됩니다.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            생성 과정을 결정적으로 만들기 위해 하나 이상의 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 리스트 입니다.
        latents (`torch.FloatTensor`, *optional*):
            사전 생성된 노이즈 잠재 변수로, 가우시안 분포에서 샘플링된 것이며, 이미지 생성을 위한 입력으로 사용됩니다.
            다양한 프롬프트로 동일한 생성을 조정하는 데 사용할 수 있습니다. 제공되지 않으면, 제공된 랜덤 `generator`를 사용해 잠재 변수 텐서가 생성됩니다.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            사전 생성된 텍스트 임베딩입니다. 텍스트 입력을 쉽게 조정하는 데 사용할 수 있습니다.(예시: 프롬프트 가중치)
            제공되지 않으면, 텍스트 임베딩이 `prompt` 입력 인수에서 생성됩니다.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            사전 생성된 네거티브 텍스트 임베딩입니다. 텍스트 입력을 쉽게 조정하는 데 사용할 수 있습니다(예시: 프롬프트 가중치).
            제공되지 않으면, `negative_prompt` 입력 인수에서 네거티브 텍스트 임베딩이 생성됩니다.
        output_type (`str`, *optional*, defaults to `"pil"`):
            생성된 이미지의 출력 형식입니다.
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` 또는 `np.array` 중에서 선택할 수 있습니다.
        return_dict (`bool`, *optional*, defaults to `True`):
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]을 반환할지, 아니면 일반 튜플을 반환할지 여부입니다.
        callback (`Callable`, *optional*):
            추론 동안 `callback_steps` 단계마다 호출될 함수입니다. 이 함수는 `callback(step: int, timestep: int, latents: torch.FloatTensor)`인수를 호출합니다.
        callback_steps (`int`, *optional*, defaults to 1):
            `callback` 함수가 호출될 빈도입니다. 지정하지 않으면, 매 단계마다 콜백이 호출됩니다.
        cross_attention_kwargs (`dict`, *optional*):
            `AttentionProcessor`에 전달될 수 있는 kwargs 딕서녀리입니다. 이는 [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)의 `self.processor`에서 정의된 대로 사용됩니다.
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)에서 제안된 리스케일 지침 요소입니다.
            `guidance_scale`은 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)에서 식 16의 `φ`로 정의됩니다.
            리스케일 지침 요소는 제로 터미널 신호 대 잡음 비(Signal Noise Ratio; SNR)을 사용할 때 과도하게 노출에 대해 수정하는 데 도움이 됩니다.

    예시:

    반환값:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] 또는 `tuple`:
        `return_dict`가 True인 경우 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]을 반환하고, 그렇지 않으면 `tuple`을 반환합니다.
        튜플을 반환할 때, 첫 번째 요소는 생성된 이미지 목록이며, 두 번째 요소는 `safety_checker`에 따라 해당 이미지가 "작업에 안전하지 않은" (nsfw) 콘텐츠일 가능성이 있는지 여부를 나타내는 `bool` 값의 목록입니다.
    """
    # 0. 기본 높이와 너비를 U-Net에 맞추어 설정
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. 입력을 확인하고, 올바르지 않으면 오류를 발생시킵니다.
    self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    # 2. 호출 파라미터를 정의합니다.
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    
    # 여기서 `guidance_scale`은 Imagen 논문(https://arxiv.org/pdf/2205.11487.pdf)의 식 (2)에서의 안내된 가중치 `w`에 유사하게 정의됩니다.
    # `guidance_scale = 1`은 분류기 없는 지침을 수행하지 않음을 의미합니다.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. 입력 프롬프트를 인코딩합니다.
    text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. 타임스텝을 준비
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. 잠재 변수를 준비
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. 추가 단계 kwargs를 준비합니다. 할일: 로직은 파이프라인에서 제거되어야 합니다.
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. 노이즈를 제거한 루프
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # 분류기 없는 가이드를 사용하는 경우, 잠재 변수를 확장.
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 노이즈 잔차 예측.
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # 지침 수행
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # 해당 논문(https://arxiv.org/pdf/2305.08891.pdf)의 3.4.를 기반
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # 이전 노이즈 샘플 x_t에서 x_t-1을 계산합니다.
            latents, log_prob = ddim_step_with_logprob(self.scheduler, noise_pred, t, latents, **extra_step_kwargs)

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # 제공된 경우, 콜백 호출.
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # 마지막 모델을 CPU로 적재.
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return image, has_nsfw_concept, all_latents, all_log_probs
