from transformers import BitsAndBytesConfig
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config
from dataclasses import dataclass, field
import torch

from finetuning_buckets.models import get_model
from finetuning_buckets.inference.safety_eval import evaluator
from datasets import disable_caching
disable_caching()


@dataclass
class ScriptArguments:

    safety_bench: str = field(default="hex-phi", metadata={"help": "the safety benchmark"})
    model_family: str = field(default="llama2", metadata={"help": "the model family"})
    prompt_style: str = field(default="llama2", metadata={"help": "the string prompt style"})
    evaluator: str = field(default="key_word", metadata={"help": "the evaluator"})
    save_path: str = field(default=None, metadata={"help": "the save path"})
    eval_template: str = field(default="plain", metadata={"help": "the eval template"})

    # 修改batch
    batch_size_per_device: int = field(default=1, metadata={"help": "the batch size"})
    max_new_tokens: int = field(default=512, metadata={"help": "the maximum number of new tokens"})
    do_sample: bool = field(default=True, metadata={"help": "do sample"})
    top_p: float = field(default=0.6, metadata={"help": "top p"})
    temperature: float = field(default=0.9, metadata={"help": "temperature"})
    use_cache: bool = field(default=True, metadata={"help": "use cache"})
    top_k: int = field(default=50, metadata={"help": "top k"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty"})
    length_penalty: float = field(default=1.0, metadata={"help": "length penalty"})

    # applied when evaluating the prefilling of a certain prefix
    prefill_prefix: str = field(default=None, metadata={"help": "the prefill prefix"})

    # applied when evaluating the prefilling of a certain number of tokens
    num_perfix_tokens: int = field(default=0, metadata={"help": "the number of prefix tokens"})


if __name__ == "__main__":

    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()


    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    
    print(f"torch_dtype: {torch_dtype}")
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        
        quantization_config=quantization_config,
    )


    ################
    # Model & Tokenizer
    ################

    model, tokenizer = get_model.get_model(model_config.model_name_or_path, model_kwargs, model_family=args.model_family, padding_side="left" )

    # 输入文本
    input_text = "今天的天气怎么样？"

    # 将输入文本编码成token
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs['input_ids']

    # 设置模型生成的最大长度
    max_length = 50  # 生成的文本最大长度，可以根据需要调整

    # 开始生成
    model.eval()
    with torch.no_grad():
        # 初始化生成过程
        generated_ids = input_ids
        for step in range(max_length):
            # 获取当前token的logits
            outputs = model(generated_ids)
            logits = outputs.logits

            # 获取当前生成步骤的最后一个token的logits
            last_token_logits = logits[0, -1, :]

            # 将logits转换为概率
            probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)

            # 获取前几个候选token的概率
            top_k = 10
            top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

            # 将这些token的id转换为实际的token
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

            # 输出当前步骤的候选token和它们的概率
            print(f"Step {step + 1}:")
            for token, prob in zip(top_k_tokens, top_k_probs):
                print(f"  Token: {token}, Probability: {prob.item()}")

            # 选择最可能的token并加入到生成的tokens中
            next_token = top_k_indices[0].item()  # 选择概率最高的token
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]])], dim=-1)

            # 如果生成了[EOS]（例如文本结束符），则提前停止生成
            if next_token == tokenizer.eos_token_id:
                break

    # 解码生成的tokens为文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nGenerated text:", generated_text)
