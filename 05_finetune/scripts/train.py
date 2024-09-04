import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
#    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch

#import bitsandbytes as bnb
#from huggingface_hub import login, HfFolder


def parse_arge():
    """인수를 파싱합니다."""
    parser = argparse.ArgumentParser()
    # 모델 ID와 데이터 세트 경로 인수 추가
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="lm_dataset", help="Path to dataset."
    )
    # parser.add_argument(
    #     "--hf_token", type=str, default=HfFolder.get_token(), help="Path to dataset."
    # )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    # parser.add_argument(
    #     "--merge_weights",
    #     type=bool,
    #     default=True,
    #     help="Whether to merge LoRA weights with base model.",
    # )
    args, _ = parser.parse_known_args()

    # if args.hf_token:
    #     print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
    #     login(token=args.hf_token)

    return args


# # https://github.com/artidoro/qlora/blob/main/qlora.py 에서 복사한 코드
# def print_trainable_parameters(model, use_4bit=False):
#     """
#     모델에서 학습 가능한 파라미터 수를 출력합니다.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         num_params = param.numel()
#         # DS Zero 3을 사용하고 가중치가 빈 상태로 초기화된 경우
#         if num_params == 0 and hasattr(param, "ds_numel"):
#             num_params = param.ds_numel
#
#         all_param += num_params
#         if param.requires_grad:
#             trainable_params += num_params
#     # if use_4bit:
#     #     trainable_params /= 2
#     print(
#         f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
#     )


# https://github.com/artidoro/qlora/blob/main/qlora.py 에서 복사한 코드
# def find_all_linear_names(model):
#     lora_module_names = set()
# #    for name, module in model.named_modules():
#         # if isinstance(module, bnb.nn.Linear4bit):
#         #     names = name.split(".")
#         #     lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#     if "lm_head" in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove("lm_head")
#     return list(lora_module_names)


# def create_peft_model(model, gradient_checkpointing=True, bf16=True):
#     from peft import (
#         get_peft_model,
#         LoraConfig,
#         TaskType,
# #        prepare_model_for_kbit_training,
#     )
#     from peft.tuners.lora import LoraLayer

#     # # 훈련을 위해 int-4 모델 준비
#     # model = prepare_model_for_kbit_training(
#     #     model, use_gradient_checkpointing=gradient_checkpointing
#     # )
#     if gradient_checkpointing:
#         model.gradient_checkpointing_enable()

#     # LoRA 대상 모듈 가져오기
#     # modules = find_all_linear_names(model)

# #     # 모델의 어텐션 블록만 대상으로 하는 경우
# #     modules = ["q_proj", "v_proj"]

# #     # 모든 선형 레이어를 대상으로 하는 경우
# #     #target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    
# #     print(f"Found {len(modules)} modules to quantize: {modules}")

#     # peft_config = LoraConfig(
#     #     r=64,
#     #     lora_alpha=16,
#     #     target_modules=modules,
#     #     lora_dropout=0.1,
#     #     bias="none",
#     #     task_type=TaskType.CAUSAL_LM,
#     # )

# #     model = get_peft_model(model, peft_config)

# #     # 'norm' 레이어를 float 32로 상향 변환하여 모델 전처리
# #     for name, module in model.named_modules():
# #         if isinstance(module, LoraLayer):
# #             if bf16:
# #                 module = module.to(torch.bfloat16)
# #         if "norm" in name:
# #             module = module.to(torch.float32)
# #         if "lm_head" in name or "embed_tokens" in name:
# #             if hasattr(module, "weight"):
# #                 if bf16 and module.weight.dtype == torch.float32:
# #                     module = module.to(torch.bfloat16)

# #     model.print_trainable_parameters()

#     return model


def training_function(args):
    # 시드 설정
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)
    # bnb 설정을 사용하여 허깅페이스 허브에서 모델 불러오기
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False
        if args.gradient_checkpointing
        else True,  # 그래디언트 체크포인팅에 필요합니다
        device_map="auto",
#        quantization_config=bnb_config,
    )

#     # PEFT 구성 생성
#     model = create_peft_model(
#         model, gradient_checkpointing=args.gradient_checkpointing, bf16=args.bf16
#     )

    # 훈련 인수 정의
    output_dir = "./tmp/llama2"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # 가능하면 BF16 사용
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        # 로깅 전략
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
    )

    # 트레이너 인스턴스 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    # 훈련 시작
    trainer.train()

    sagemaker_save_dir="/opt/ml/model/"
#     if args.merge_weights:
#         # 어댑터 가중치를 기본 모델과 병합하고 저장
#         # int-4 모델 저장
#         trainer.model.save_pretrained(output_dir, safe_serialization=False)
#         # 메모리 정리
#         del model
#         del trainer
#         torch.cuda.empty_cache()

#         from peft import AutoPeftModelForCausalLM

#         # fp16에서 PEFT 모델 로드
#         model = AutoPeftModelForCausalLM.from_pretrained(
#             output_dir,
#             low_cpu_mem_usage=True,
#             torch_dtype=torch.bfloat16,
#         )
#         # LoRA와 기본 모델을 병합하고 저장
#         model = model.merge_and_unload()
#         model.save_pretrained(
#             sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB"
#         )
    # else:
    trainer.model.save_pretrained(
        sagemaker_save_dir, safe_serialization=True
    )

    # 쉽게 추론할 수 있도록 토크나이저 저장
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(sagemaker_save_dir)


def main():
    args = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()