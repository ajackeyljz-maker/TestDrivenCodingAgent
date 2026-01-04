import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float16 if use_cuda else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=dtype
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "用10种不同的计算机语言快速排序"
messages = [
    {"role": "system", "content": "你是一位强大的代码助手."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)