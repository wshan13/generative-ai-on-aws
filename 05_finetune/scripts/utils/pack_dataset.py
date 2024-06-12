from functools import partial
from itertools import chain


# 다음 배치에서 사용할 수 있도록 배치에서 남은 부분을 저장할 빈 리스트 생성
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


def pack_dataset(dataset, chunk_length=2048):
    print(f"Chunking dataset into chunks of {chunk_length} tokens.")

    def chunk(sample, chunk_length=chunk_length):
        # 배치에서 남은 데이터를 다음 배치에서 재사용하기 위한 전역 변수 정의
        global remainder
        # 모든 텍스트를 연결하고 이전 배치에서 남은 데이터를 추가
        concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
        concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
        # 배치의 총 토큰 수 계산
        batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

        # 배치의 최대 청크 수 가져오기
        if batch_total_length >= chunk_length:
            batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

        # max_len 크기로 청크 나누기
        result = {
            k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
            for k, t in concatenated_examples.items()
        }
        # 남은 데이터를 다음 배치를 위한 전역 변수에 추가
        remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
        # 레이블 준비
        result["labels"] = result["input_ids"].copy()
        return result

    # 데이터세트를 토크나이즈하고 청크 나누기
    lm_dataset = dataset.map(
        partial(chunk, chunk_length=chunk_length),
        batched=True,
    )
    print(f"Total number of samples: {len(lm_dataset)}")
    return lm_dataset