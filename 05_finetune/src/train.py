import argparse
import os
import json
import pprint

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, GenerationConfig
from datasets import load_dataset

def list_files(startpath):
    """디렉터리 내 파일 목록을 표시하는 도우미 함수"""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation_data", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--checkpoint_base_path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--validation_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--train_sample_percentage", type=float, default=0.01)
    parser.add_argument("--model_checkpoint", type=str, default=None)    
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])  # This is unused
    
    args, _ = parser.parse_known_args()
    print("Args:")
    print(args)

    env_var = os.environ
    print("Environment Variables:")
    pprint.pprint(dict(env_var), width=1)

    return args


if __name__ == "__main__":
    
    # 인수 파싱
    args = parse_args()
    
    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    
    # 입력 파일 탐색
    local_data_processed_path = '/opt/ml/input/data'
    print('Listing all input data files...')
    list_files(local_data_processed_path)
    
    # 데이터 세트 로드
    print(f'loading dataset from: {local_data_processed_path}')
    tokenized_dataset = load_dataset(
        local_data_processed_path,
        data_files={'train': 'train/*.parquet', 'test': 'test/*.parquet', 'validation': 'validation/*.parquet'}
    ).with_format("torch")
    print(f'loaded dataset: {tokenized_dataset}')
    
    # 학습을 위한 데이터 세트 샘플링
    skip_inds = int(1 / args.train_sample_percentage)
    sample_tokenized_dataset = tokenized_dataset.filter(lambda example, indice: indice % skip_inds == 0, with_indices=True)

    # 모델 학습
    output_dir = args.checkpoint_base_path
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.validation_batch_size,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sample_tokenized_dataset['train'],
        eval_dataset=sample_tokenized_dataset['validation']
    )
    trainer.train()
    
    # 모델 저장
    transformer_fine_tuned_model_path = os.environ["SM_MODEL_DIR"]
    os.makedirs(transformer_fine_tuned_model_path, exist_ok=True)
    print(f"Saving the final model to: transformer_fine_tuned_model_path={transformer_fine_tuned_model_path}")
    model.save_pretrained(transformer_fine_tuned_model_path)
    tokenizer.save_pretrained(transformer_fine_tuned_model_path)
    
    # 모델 추론을 위해 inference.py와 requirements.txt를 code/ 디렉터리로 복사
    #   참고: 이것은 세이지메이커 엔드포인트가 해당 파일을 인식하기 위해 필요한 코드 입니다.
    #        하드코딩 되어있어서 반드시 code/ 디렉터리로 불러와야 합니다.
    local_model_dir = os.environ["SM_MODEL_DIR"]
    inference_path = os.path.join(local_model_dir, "code/")
    print("Copying inference source files to {}".format(inference_path))
    os.makedirs(inference_path, exist_ok=True)
    os.system("cp inference.py {}".format(inference_path))
    os.system('cp requirements.txt {}'.format(inference_path))
    print(f'Files in inference code path "{inference_path}"')
    list_files(inference_path)
