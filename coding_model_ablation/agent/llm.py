from __future__ import annotations
import json
from typing import Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# 接口约定 load_llm的返回值被标注为LLM 必须有name和generate
class LLM(Protocol):
    name: str
    def generate(self, prompt: str) -> str: ...

# 输出是JSON格式直接返回 不是泽保留原文本
def _extract_json_text(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text.strip()
    candidate = text[start: end + 1]
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return text.strip()


class TransformersLLM:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ):
        self.name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        # 量化参数
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=quant_config,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # 设置pad token id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        # 对话模版 chat template
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            # 自动在末尾加上模型回答标记 让模型开始回答
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_ids = input_ids.to(self.model.device)
        do_sample = self.temperature > 0

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        """
            input/output.shape = [batch, seq_len] 这里的 batch = 1
            且output seq_len是模型生成的完整序列 包含prompt和生成序列 
            input_ids.shape[-1]是prompt的长度 这里切片获取生成序列
        """
        gen_ids = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # 尝试抽取并返回 JSON 字符串片段
        return _extract_json_text(text)  


def load_llm(model_name: str) -> LLM:
    return TransformersLLM(model_name)
