import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### 일반적인 파라미터 ######
    # wandb 로깅 및 체크포인트 저장을 위한 실행 이름: 제공되지 않으면 날짜와 시간을 기반으로 자동 생성됩니다.
    config.run_name = ""
    # 재현성을 위한 랜덤 시드
    config.seed = 42
    # 체크포인트 저장을 위한 최상위 로깅 경로
    config.logdir = "logs"
    # 추론을 위한 출력 모델 경로
    config.out_dir = "trained_model"
    # 학습할 에포크 수: 각 에포크는 모델에서 샘플링한 후 그 샘플들로 학습하는 한 라운드입니다.
    config.num_epochs = 10  #  100
    # 모델 체크포인트 저장 간격
    config.save_freq = 20
    # 오래된 체크포인트를 덮어쓰기 전에 유지할 체크포인트 수
    config.num_checkpoint_limit = 5
    # 혼합 정밀도 학습: 선택 옵션은 "fp16", "bf16", 및 "no"입니다. 반 정밀도는 학습 속도를 크게 향상시킵니다.
    config.mixed_precision = "fp16"
    # Ampere GPU에서 TF32 허용: 학습 속도를 높일 수 있습니다.
    config.allow_tf32 = True
    # 체크포인트에서부터 학습 재개. 정확한 체크포인트 경로(예: checkpoint_50) 또는 체크포인트를 포함하는 경로를 지정합니다.
    # 이 경우 최신 체크포인트가 사용됩니다. `config.use_lora`는 저장된 체크포인트를 생성할 실행과 동일한 값으로 설정해야 합니다.
    config.resume_from = ""
    # 저순위 적응 사용 여부. 저순위 적응은 U-Net의 어텐션 레이어에 작은 가중치 행렬을 주입해 메모리 사용을 크게 줄입니다.
    # 저순위 적응과 fp16, 배치 크기 1을 사용하면 스테이블 디퓨전의 미세 조정은 약 10GB의 GPU 메모리를 필요로 합니다.
    # 저순위 적응이 비활성화되면 학습하는데 많은 메모리가 소모되며 저장될 체크포인트 파일도 커집니다.
    config.use_lora = True

    ###### 사전 학습된 모델 ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # 적재될 기본 모델: 로컬 경로 또는 허깅 페이스 모델 허브에서 모델 이름을 지정합니다.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    # 적재되 모델 버전
    pretrained.revision = "main"

    ###### 샘플링 ######
    config.sample = sample = ml_collections.ConfigDict()
    # 샘플러 추론 단계 수
    sample.num_steps = 50  # 50
    # DDIM 샘플러의 eta 파라미터. 샘플링 과정에 주입되는 노이즈의 양을 제어합니다.
    # 0.0은 완전히 결정적이며, 1.0은 DDPM 샘플러와 동일합니다.
    sample.eta = 1.0
    # 분류기 없이 안내(classifier-free guidance) 가중치: 1.0은 분류기를 활용하지 않는 경우입니다.
    sample.guidance_scale = 5.0
    # 샘플링에 사용할 배치 크기 (GPU 당!)
    sample.batch_size = 1
    # 에포크당 샘플링 배치 수: 총 샘플 수는 `num_batches_per_epoch * batch_size * num_gpus`입니다.
    sample.num_batches_per_epoch = 2
    
    ###### 정답 보상 #####
    config.sns_topic_arn = "arn:aws:sns:us-east-1:463383979161:create-rewards"
    config.sqs_name = "rewards"
    config.sqs_region = "us-east-1"
    config.image_dir = "images"
    
    ###### 학습 ######
    config.train = train = ml_collections.ConfigDict()
    # 학습에 사용할 배치 크기 (GPU 당!)
    train.batch_size = 1
    # bitsandbytes의 8비트 아담 옵티마이저 사용 여부
    train.use_8bit_adam = False
    # 학습률
    train.learning_rate = 3e-4
    # 아담 베타1.
    train.adam_beta1 = 0.9
    # 아담 베타2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    # 아담 가중치 감쇠(weight decay).
    train.adam_weight_decay = 1e-4
    # 아담 엡실론.
    train.adam_epsilon = 1e-8
    # 그래디언트 누적 단계 수: 효과적인 배치 크기는 `batch_size * num_gpus * gradient_accumulation_steps`입니다.
    train.gradient_accumulation_steps = 1
    # 그래디언트 클리핑(gradient clipping)을 위한 최대 그래디언트 노름(norm).
    train.max_grad_norm = 1.0
    # 외부 에포크당 내부 에포크 수: 각 내부 에포크는 하나의 외부 에포크의 샘플링 라운드 동안 수집된 데이터를 통한 반복입니다.
    train.num_inner_epochs = 1
    # 학습 중에 분류기 없이 안내 사용 여부: 활성화되면 샘플링 중 사용된 것과 동일한 분류기 없이 안내 규모를 사용됩니다.
    train.cfg = True
    # [-adv_clip_max, adv_clip_max] 범위로 클리핑 이용.
    train.adv_clip_max = 5
    # 근접 정책 최적화 클리핑 범위
    train.clip_range = 1e-4
    # 학습할 타임스텝 비율. 1.0보다 낮게 설정하면 각 샘플의 타임스텝 하위 집합에서 모델을 학습합니다.
    # 이는 학습 속도를 높이지만 정책 그래디언트 추정에 대한 정확도를 낮출 수 있습니다.
    train.timestep_fraction = 1.0

    ###### 프롬프트 함수 ######
    # 사용할 프롬프트 함수. 사용 가능한 프롬프트 함수는 `prompts.py`를 참조하세요.
    config.prompt_fn = "simple_prompts" # "imagenet_animals"
    # 프롬프트 함수에 전달할 kwargs
    config.prompt_fn_kwargs = {}

    ###### 보상 함수 ######
    # 사용할 보상 함수: 사용 가능한 보상 함수는 `rewards.py`를 참조하세요.
    config.reward_fn = "random_score" # "aesthetic_score" # "jpeg_compressibility"

    ###### 프롬프트별 통계 추적 ######
    # 해당 파라미터가 활성화되면 모델은 프롬프트별로 보상의 평균과 표준 편차를 추적하고 이를 사용해 이점을 계산합니다.
    # 프롬프트별 통계 추적을 비활성화하려면 `config.per_prompt_stat_tracking`을 None으로 설정하며, 전체 배치의 평균과 표준 편차를 사용해 이점이 계산됩니다.
    config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    # 각 프롬프트에 대해 버퍼에 저장할 보상 값 수. 버퍼는 에포크를 넘어 지속됩니다.
    config.per_prompt_stat_tracking.buffer_size = 16
    # 프롬프트별 평균 및 표준 편차를 사용하기 전에 버퍼에 저장할 보상 값의 최소 수.
    # 버퍼에 `min_count` 값보다 적은 값이 포함된 경우, 전체 배치의 평균과 표준 편차가 대신 사용됩니다.
    config.per_prompt_stat_tracking.min_count = 16

    return config
