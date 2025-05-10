import json
from transformers import AutoTokenizer
def get_hex_phi_with_harmful_prefix(tokenizer, system_prompt = None, input_template = None, prompt_style = 'llama2', num_prefix_tokens = 0):

    if prompt_style not in ['llama2', 'llama2_base', 'gemma', 'gemma_base']:
        raise ValueError(f"prompt_style {prompt_style} not maintained in this dataset")
    
    data_file_path = '/root/autodl-tmp/safeDecoding/finetuning_buckets/datasets/data/safety_bench/Harmful-HEx-PHI.jsonl'
    output_file_path = f'/root/autodl-tmp/safeDecoding/finetuning_buckets/inference/safety_eval/Harmful-HEx-PHI_with_prefix{num_prefix_tokens}tokens_llama2.json'

    dataset = []

    with open(data_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)

    conversation_data = []
    json_data = []
    
    for idx, item in enumerate(dataset):
        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append({'role': 'user', 'content': input_template % item[0]['content']})
        else:
            messages.append({'role': 'user', 'content': item[0]['content']})

        prefix = tokenizer.encode(item[1]['content'])[1:num_prefix_tokens]
        prefix = tokenizer.decode(prefix)
        target = item[1]['content']
        goal = item[0]['content']
        messages.append({'role': 'assistant', 'content': prefix}) 
        
        conversation_data.append(messages)

        json_item = {
            "id": idx,
            "real_id": idx + 1,
            "goal": goal,
            "target": target,
            "prompt": "<s>[INST] <<SYS>>\n\n<</SYS>>\n\n"+goal+" [/INST] "+prefix,
            "response": None,
            "source": "Harmful-HEx-PHI",
            "num_steps": 1,
            "target-model": "llama2"
        }
        json_data.append(json_item)
    
    # Write the JSON data to file as a single JSON array
    with open(output_file_path, 'w') as f:
        json.dump(json_data, f, indent=2)
            
    return conversation_data, dataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/shallow-vs-deep-alignment/ckpts/Llama-2-7b-chat-fp16")
    conversation_data, dataset = get_hex_phi_with_harmful_prefix(tokenizer, system_prompt = None, input_template = None, prompt_style = 'llama2', num_prefix_tokens = 40)
