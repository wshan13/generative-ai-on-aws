import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # 활용한 DGX 기계는 8개의 GPU가 있었으므로, 이는 8 * 8 * 4 = 256 샘플을 에포크당 생성하는 것과 동일합니다.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # 이는 (8 * 4) / (4 * 2) = 4 그래디언트 업데이트를 에포크당 수행하는 것에 해당합니다.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # 프롬프팅
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # 보상
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config


def incompressibility():
    config = compressibility()
    config.reward_fn = "jpeg_incompressibility"
    return config


def aesthetic():
    config = compressibility()
    config.num_epochs = 200
    config.reward_fn = "aesthetic_score"

    # 이 보상은 최적화하기가 좀 더 어렵기 때문에, 에포크당 2번의 그래디언트 업데이트를 사용했습니다.
    config.train.gradient_accumulation_steps = 4

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config


def prompt_image_alignment():
    config = compressibility()

    config.num_epochs = 200
    # 이번 실험에서는 언어-이미지 시각 어시스턴트 추론을 위해 2개의 GPU를 예약했으므로, 노이즈 제거 확산 정책 최적화에는 6개의 GPU만 사용할 수 있습니다.
    # 에포크당 총 샘플 수는 8 * 6 * 6 = 288입니다.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 6

    # 다시 말하지만, 이것은 최적화하기가 더 어려워서 (8 * 6) / (4 * 6) = 2 그래디언트 업데이트를 에포크당 사용했습니다.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 6

    # 프롬프팅
    config.prompt_fn = "nouns_activities"
    config.prompt_fn_kwargs = {
        "nouns_file": "simple_animals.txt",
        "activities_file": "activities.txt",
    }

    # 보상
    config.reward_fn = "llava_bertscore"

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def get_config(name):
    return globals()[name]()
