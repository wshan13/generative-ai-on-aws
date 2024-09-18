import json
import logging
from typing import Any
from typing import Dict
from typing import Union
import subprocess
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from sagemaker_inference import encoder
from transformers import TextGenerationPipeline
from transformers import pipeline
from transformers import set_seed


APPLICATION_X_TEXT = "application/x-text"
APPLICATION_JSON = "application/json"
STR_DECODE_CODE = "utf-8"

VERBOSE_EXTENSION = ";verbose"

TEXT_GENERATION = "text-generation"

GENERATED_TEXT = "generated_text"
GENERATED_TEXTS = "generated_texts"

# 가능한 모델 파라미터
TEXT_INPUTS = "text_inputs"
MAX_LENGTH = "max_length"
NUM_RETURN_SEQUENCES = "num_return_sequences"
NUM_NEW_TOKENS = "num_new_tokens"
NUM_BEAMS = "num_beams"
TOP_P = "top_p"
EARLY_STOPPING = "early_stopping"
DO_SAMPLE = "do_sample"
NO_REPEAT_NGRAM_SIZE = "no_repeat_ngram_size"
TOP_K = "top_k"
TEMPERATURE = "temperature"
SEED = "seed"

ALL_PARAM_NAMES = [
    TEXT_INPUTS,
    MAX_LENGTH,
    NUM_NEW_TOKENS,
    NUM_RETURN_SEQUENCES,
    NUM_BEAMS,
    TOP_P,
    EARLY_STOPPING,
    DO_SAMPLE,
    NO_REPEAT_NGRAM_SIZE,
    TOP_K,
    TEMPERATURE,
    SEED,
]


# 모델 파라미터 범위
MAX_LENGTH_MIN = 1
NUM_RETURN_SEQUENCE_MIN = 1
NUM_BEAMS_MIN = 1
TOP_P_MIN = 0
TOP_P_MAX = 1
NO_REPEAT_NGRAM_SIZE_MIN = 1
TOP_K_MIN = 0
TEMPERATURE_MIN = 0




def model_fn(model_dir: str) -> list:
    """모델에 대한 추론 작업을 대리로 생성합니다.

    작업자 한 명당 한 번만 실행됩니다.

    Args:
        model_dir (str): 모델 파일이 저장된 디렉터리
    Returns:
        list: 허깅 페이스 토크나이저와 모델
    """
    
    print('walking model_dir: {}'.format(model_dir))

    import os
    for root, dirs, files in os.walk(model_dir, topdown=False):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))
            
    # 토크나이저와 모델 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    print(f'Loaded Local HuggingFace Tokenzier:\n{tokenizer}')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    print(f'Loaded Local HuggingFace Model:\n{model}')
    
    return [tokenizer, model]


def _validate_payload(payload: Dict[str, Any]) -> None:
    """입력 로드의 파라미터를 검증합니다.

    max_length, num_return_sequences, num_beams, top_p, temprature 값이 유효한 범위에 있는지 확인합니다.
    do_sample 값이 boolean인지 확인합니다.
    max_length, num_return_sequences, num_beams, seed 값이 정수인지 확인합니다.

    Args:
        payload: 디코딩된 입력 페이로드 (입력 파라미터와 값을 가진 딕셔너리)
    """
    for param_name in payload:
        assert (
            param_name in ALL_PARAM_NAMES
        ), f"Input payload contains an invalid key {param_name}. Valid keys are {ALL_PARAM_NAMES}."

    assert TEXT_INPUTS in payload, f"Input payload must contain {TEXT_INPUTS} key."

    for param_name in [MAX_LENGTH, NUM_RETURN_SEQUENCES, NUM_BEAMS, SEED]:
        if param_name in payload:
            assert type(payload[param_name]) == int, f"{param_name} must be an integer, got {payload[param_name]}."

    if MAX_LENGTH in payload:
        assert (
            payload[MAX_LENGTH] >= MAX_LENGTH_MIN
        ), f"{MAX_LENGTH} must be at least {MAX_LENGTH_MIN}, got {payload[MAX_LENGTH]}."
    if NUM_RETURN_SEQUENCES in payload:
        assert payload[NUM_RETURN_SEQUENCES] >= NUM_RETURN_SEQUENCE_MIN, (
            f"{NUM_RETURN_SEQUENCES} must be at least {NUM_RETURN_SEQUENCE_MIN}, "
            f"got {payload[NUM_RETURN_SEQUENCES]}."
        )
    if NUM_BEAMS in payload:
        assert (
            payload[NUM_BEAMS] >= NUM_BEAMS_MIN
        ), f"{NUM_BEAMS} must be at least {NUM_BEAMS_MIN}, got {payload[NUM_BEAMS]}."
    if NUM_RETURN_SEQUENCES in payload and NUM_BEAMS in payload:
        assert payload[NUM_RETURN_SEQUENCES] <= payload[NUM_BEAMS], (
            f"{NUM_BEAMS} must be at least {NUM_RETURN_SEQUENCES}. Instead got "
            f"{NUM_BEAMS}={payload[NUM_BEAMS]} and {NUM_RETURN_SEQUENCES}="
            f"{payload[NUM_RETURN_SEQUENCES]}."
        )
    if TOP_P in payload:
        assert TOP_P_MIN <= payload[TOP_P] <= TOP_P_MAX, (
            f"{TOP_K} must be in range [{TOP_P_MIN},{TOP_P_MAX}], got "
            f"{payload[TOP_P]}"
        )
    if TEMPERATURE in payload:
        assert payload[TEMPERATURE] >= TEMPERATURE_MIN, (
            f"{TEMPERATURE} must be a float with value at least {TEMPERATURE_MIN}, got "
            f"{payload[TEMPERATURE]}."
        )
    if DO_SAMPLE in payload:
        assert (
            type(payload[DO_SAMPLE]) == bool
        ), f"{DO_SAMPLE} must be a boolean, got {payload[DO_SAMPLE]}."


def _update_num_beams(payload: Dict[str, Union[str, float, int]]) -> Dict[str, Union[str, float, int]]:
    """num_return_sequences가 존재하고 num_beams가 누락된 경우, num_beams를 페이로드에 추가합니다."""
    if NUM_RETURN_SEQUENCES in payload and NUM_BEAMS not in payload:
        payload[NUM_BEAMS] = payload[NUM_RETURN_SEQUENCES]
    return payload


def transform_fn(model_objs: list, input_data: bytes, content_type: str, accept: str) -> bytes:
    """모델에 대한 예측을 수행하고 직렬화된 응답을 반환합니다.

    함수 서명은 세이지메이커 계약에 부합합니다.

    Args:
        model_objs (list): 토크나이저, 모델
        input_data (obj): 요청 데이터
        content_type (str): 요청 내용 유형
        accept (str): 클라이언트가 예상하는 받아들여질 헤더
    Returns:
        obj: 예측의 바이트 문자열
    """
    tokenizer = model_objs[0]
    model = model_objs[1]
    
    if content_type == APPLICATION_X_TEXT:
        try:
            input_text = input_data.decode(STR_DECODE_CODE)
        except Exception:
            logging.exception(
                f"Failed to parse input payload. For content_type={APPLICATION_X_TEXT}, input "
                f"payload must be a string encoded in utf-8 format."
            )
            raise
        try:
            # output = text_generator(input_text)[0]
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            original_outputs = model.generate(input_ids,
                                              GenerationConfig(max_new_tokens=200)
                                             )
            output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
        except Exception:
            logging.exception("Failed to do inference")
            raise

    # TODO: 더 많은 추론 옵션을 위해 여기에 JSON 구현 통합 필요
    # elif content_type == APPLICATION_JSON:
    #     try:
    #         payload = json.loads(input_data)
    #     except Exception:
    #         logging.exception(
    #             f"Failed to parse input payload. For content_type={APPLICATION_JSON}, input "
    #             f"payload must be a json encoded dictionary with keys {ALL_PARAM_NAMES}."
    #         )
    #         raise
    #     _validate_payload(payload)
    #     payload = _update_num_beams(payload)
    #     if SEED in payload:
    #         set_seed(payload[SEED])
    #         del payload[SEED]
    #     try:
    #         model_output = text_generator(**payload)
    #         input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    #         original_outputs = model.generate(input_ids,
    #                                           GenerationConfig(max_new_tokens=200)
    #                                          )
    #         model_output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
    #         output = {GENERATED_TEXTS: [x[GENERATED_TEXT] for x in model_output]}
    #     except Exception:
    #         logging.exception("Failed to do inference")
    #         raise

    else:
        raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
    if accept.endswith(VERBOSE_EXTENSION):
        accept = accept.rstrip(VERBOSE_EXTENSION)  # Verbose and non-verbose response are identical
    return encoder.encode(output, accept)
