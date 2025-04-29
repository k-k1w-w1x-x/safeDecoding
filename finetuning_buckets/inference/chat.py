import torch
import time

class Chat:
    # by default, will maintain all conversations in OpenAI chat format

    def __init__(self, model, prompt_style, tokenizer, max_length=2048, 
                 init_conversation = None, init_system_prompt = None):
        
        if init_conversation is not None and init_system_prompt is not None:
            raise ValueError("init_conversation and init_system_prompt cannot be provided at the same time")
       
        self.model = model
        self.prompt_style = prompt_style
        self.tokenizer = tokenizer
        self.max_length = max_length # limit the length of the whole conversation
        

        # formatter will be used to convert openai chat format to string
        if prompt_style == 'llama2':
            from finetuning_buckets.models.model_families.llama2 import LlamaStringConverter, default_system_prompt
            self.string_formatter = LlamaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        elif prompt_style == 'gemma':
            from finetuning_buckets.models.model_families.gemma import GemmaStringConverter, default_system_prompt
            self.string_formatter = GemmaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        elif prompt_style == 'llama2_base':
            from finetuning_buckets.models.model_families.llama2_base import LlamaStringConverter, default_system_prompt, base_stopping_criteria
            self.string_formatter = LlamaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = base_stopping_criteria
        elif prompt_style == 'gemma_base':
            from finetuning_buckets.models.model_families.gemma_base import GemmaStringConverter, default_system_prompt, base_stopping_criteria
            self.string_formatter = GemmaStringConverter.string_formatter_completion_only
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = base_stopping_criteria
        else:
            raise ValueError(f"Prompt style {prompt_style} not supported")


        if init_conversation is not None:
            self.validate_conversation(init_conversation)
            if isinstance(init_conversation, dict):
                init_conversation = init_conversation['messages']

            if init_conversation[-1]['role'] == 'user':
                raise ValueError("the last message of init_conversation should be assistant message or system prompt, not user message")

            if init_conversation[0]['role'] != 'system':
                self.system_prompt = self.default_system_prompt
                self.converstaion = self.init_conversation() + init_conversation
            else:
                self.system_prompt = init_conversation[0]['content']
                self.converstaion = init_conversation
        else:

            if init_system_prompt is not None:
                self.system_prompt = init_system_prompt
            else:
                self.system_prompt = self.default_system_prompt
            
            self.converstaion = self.init_conversation()

        

    def __call__(self, text, max_new_tokens = 1024, 
                 do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                 repetition_penalty = 1.0, length_penalty = 1.0, **kwargs):
        
        self.update_conversation(user_message=text)
        _, tokens_input = self.prepare_model_input(conversation=self.conversation, max_new_tokens=max_new_tokens)

        tokens_input = tokens_input.unsqueeze(0).to(self.model.device)
        input_length = tokens_input.shape[1]
        outputs = self.model.generate(
                input_ids = tokens_input,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                stopping_criteria = self.stopping_criteria,
                **kwargs
            )

        output_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        self.update_conversation(assistant_message=output_text)

        return output_text
    

    
    def generate_one_shot(self, input, max_new_tokens = 1024, 
                 do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                 repetition_penalty = 1.0, length_penalty = 1.0, **kwargs):
        # a one-shot conversation, input can be a string, a list of messages, or a dictionary with 'messages' key
        # no history will be maintained for one-shot conversation
        
        if isinstance(input, dict) or isinstance(input, list):
            input = self.validate_conversation(input)
        elif isinstance(input, str):
            input = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
        else:
            raise ValueError(f"input {input} is not a valid conversation input")

        

        _, tokens_input = self.prepare_model_input(input, max_new_tokens)
        tokens_input = tokens_input.to(self.model.device)
        input_length = tokens_input.shape[1]


        outputs = self.model.generate(
                input_ids = tokens_input,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                stopping_criteria = self.stopping_criteria,
                **kwargs
            )

        output_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True) # the model output part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True) # the whole conversation

        return output_text, full_text
    

    def generate_one_shot_in_batch(self, inputs, accelerator, max_new_tokens = 1024, 
                 do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                 repetition_penalty = 1.0, length_penalty = 1.0, **kwargs):
        # a one-shot conversation, input can be a string, a list of messages, or a dictionary with 'messages' key
        # no history will be maintained for one-shot conversation
        # this function is for batch inference to accelerate the evaluation
        
        inputs_processed = []

        for item in inputs:

            if isinstance(item, dict) or isinstance(item, list):
                item_processed = self.validate_conversation(item)
            elif isinstance(item, str):
                item_processed = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
            else:
                raise ValueError(f"input {item} is not a valid conversation input")
            
            item_processed = self.string_formatter({'messages': item_processed})['text']

            inputs_processed.append(item_processed)
        

        model_inputs = self.tokenizer(inputs_processed, padding = True, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
                input_ids = model_inputs['input_ids'],
                attention_mask = model_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                stopping_criteria = self.stopping_criteria,
                **kwargs
            )

        full_texts = [] # the whole conversation texts
        output_texts = [] # the model output part texts

        for i, item in enumerate(outputs):

            input_pos = model_inputs['attention_mask'][i].nonzero()

            input_length = input_pos.shape[0] # how many input tokens
            start_pos = input_pos[0][0] # the first non-padding token

            full_text = self.tokenizer.decode(item, skip_special_tokens=True)
            output_text = self.tokenizer.decode(item[start_pos + input_length:], skip_special_tokens=True)

            full_texts.append(full_text)
            output_texts.append(output_text)

        
        return output_texts, full_texts
    def validate_conversation(self, conversation=None):
        # validate the conversation format, return the conversation in OpenAI chat format

        if conversation is None:
            conversation = self.conversation

        if isinstance(conversation, dict):
            if 'messages' not in conversation:
                raise ValueError(f"conversation {conversation} does not have 'messages' key")
            convs = conversation['messages']

        else: 
            convs = conversation
        
        if not isinstance(convs, list):
            raise ValueError(f"conversation {conversation} is not a valid list of messages")

        if len(convs) == 0:
            raise ValueError(f"the conversation {conversation} is empty")
        
        for conv in convs:
            if 'role' not in conv or 'content' not in conv:
                raise ValueError(f"the message {conv} does not have 'role' or 'content' key")

        
        if convs[0]['role'] != 'system':
            convs = self.init_conversation() + convs

        pt = 1
        
        while pt < len(convs):
            if convs[pt]['role'] != 'user':
                raise ValueError(f"the message should be user - assistant alternation, but the {pt}th message is {convs[pt]['role']}")
            pt += 1
            if pt >= len(convs):
                break
            if convs[pt]['role'] != 'assistant':
                raise ValueError(f"the message should be user - assistant alternation, but the {pt}th message is {convs[pt]['role']}")
            pt += 1
        return convs
    
    def init_conversation(self, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        return [{'role': 'system', 'content': system_prompt}]
    
    def refresh_conversation(self):
        self.conversation = self.init_conversation()
    
    def update_conversation(self, conversation = None, user_message=None, assistant_message=None):
        if conversation is None:
            conversation = self.conversation
        
        if user_message is None and assistant_message is None:
            raise ValueError("user_message or assistant_message should be provided")
        
        if user_message is not None:
            if conversation[-1]['role'] == 'user':
                raise ValueError("the message should be user - assistant alternation")
            conversation.append({'role': 'user', 'content': user_message})
        
        if assistant_message is not None:
            if conversation[-1]['role'] == 'assistant' or conversation[-1]['role'] == 'system':
                raise ValueError("the message should be user - assistant alternation")
            conversation.append({'role': 'assistant', 'content': assistant_message})
    
    def prepare_model_input(self, conversation=None, max_new_tokens=512):
        if conversation is None:
            conversation = self.conversation
        string_input = self.string_formatter({'messages': conversation})['text']
        
        tokens_input = self.tokenizer.encode(string_input, return_tensors="pt", 
                                             max_length=self.max_length - max_new_tokens, truncation=True)

        return string_input, tokens_input
class SafeChat(Chat):
    def __init__(self, model,expert_model, prompt_style, tokenizer, max_length=2048, 
                 init_conversation = None, init_system_prompt = None):
        
        if init_conversation is not None and init_system_prompt is not None:
            raise ValueError("init_conversation and init_system_prompt cannot be provided at the same time")
       
        self.model = model
        self.expert_model = expert_model
        self.prompt_style = prompt_style
        self.tokenizer = tokenizer
        self.max_length = max_length # limit the length of the whole conversation
        

        # formatter will be used to convert openai chat format to string
        if prompt_style == 'llama2':
            from finetuning_buckets.models.model_families.llama2 import LlamaStringConverter, default_system_prompt
            self.string_formatter = LlamaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        elif prompt_style == 'gemma':
            from finetuning_buckets.models.model_families.gemma import GemmaStringConverter, default_system_prompt
            self.string_formatter = GemmaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        elif prompt_style == 'llama2_base':
            from finetuning_buckets.models.model_families.llama2_base import LlamaStringConverter, default_system_prompt, base_stopping_criteria
            self.string_formatter = LlamaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = base_stopping_criteria
        elif prompt_style == 'gemma_base':
            from finetuning_buckets.models.model_families.gemma_base import GemmaStringConverter, default_system_prompt, base_stopping_criteria
            self.string_formatter = GemmaStringConverter.string_formatter_completion_only
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = base_stopping_criteria
        else:
            raise ValueError(f"Prompt style {prompt_style} not supported")


        if init_conversation is not None:
            self.validate_conversation(init_conversation)
            if isinstance(init_conversation, dict):
                init_conversation = init_conversation['messages']

            if init_conversation[-1]['role'] == 'user':
                raise ValueError("the last message of init_conversation should be assistant message or system prompt, not user message")

            if init_conversation[0]['role'] != 'system':
                self.system_prompt = self.default_system_prompt
                self.converstaion = self.init_conversation() + init_conversation
            else:
                self.system_prompt = init_conversation[0]['content']
                self.converstaion = init_conversation
        else:

            if init_system_prompt is not None:
                self.system_prompt = init_system_prompt
            else:
                self.system_prompt = self.default_system_prompt
            
            self.converstaion = self.init_conversation()
    def safe_decoding(self,tensor1, tensor2,C,alpha):
        tensor2=tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        left, right = 1, n
        ans_k = None
        while left <= right:
            mid = (left + right) // 2
            indices1 = set(torch.topk(tensor1, mid).indices.tolist())
            indices2 = set(torch.topk(tensor2, mid).indices.tolist())
            if len(indices1 & indices2) >= C:
                ans_k = mid
                right = mid - 1
            else:
                left = mid + 1

        if ans_k is None:
            return None, None  # 若没有满足条件的 k

        # 获取最终的交集
        topk1 = torch.topk(tensor1, ans_k)
        topk2 = torch.topk(tensor2, ans_k)
        # print(f"Final top {ans_k} values for tensor1: {topk1.values},indices:{topk1.indices}")
        # print(f"Final top {ans_k} values for tensor2: {topk2.values},indices:{topk2.indices}")
        indices1 = set(topk1.indices.tolist())
        indices2 = set(topk2.indices.tolist())
        intersection = indices1 & indices2
        # print(tensor1.shape,tensor2.shape)
        # 在交集内，找到 (tensor1 - tensor2) 差值最大的下标
        max_index = max(intersection, key=lambda idx: tensor1[idx] +  alpha*(tensor2[idx] - tensor1[idx]))
        # if max_index != topk1.indices[0]:
        #     print(f"max_index is {max_index},prob1 is {tensor1[max_index]},prob2 is {tensor2[max_index]},while topk1.indices[0] is {topk1.indices[0]},prob1 is {tensor1[topk1.indices[0]]},prob2 is {tensor2[topk1.indices[0]]}")
        #     print(f"max_index value = {tensor1[max_index] +  alpha*(tensor2[max_index] - tensor1[max_index])},while topk1.indices[0] value = {tensor1[topk1.indices[0]] +  alpha*(tensor2[topk1.indices[0]] - tensor1[topk1.indices[0]])}")
        return ans_k, max_index
    def safe_decoding_new(self,tensor1, tensor2,C,alpha):
        tensor2=tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        topk1 = torch.topk(tensor1, C)
        indices1 = set(topk1.indices.tolist())
        indices2 = set([i for i in range(n)])
        intersection = indices1 & indices2
        max_index = max(intersection, key=lambda idx: tensor1[idx] +  alpha*(tensor2[idx] - tensor1[idx]))
        return C, max_index
    
    def safe_decoding_new2(self,tensor1, tensor2,C,alpha):
        tensor2=tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        left, right = 1, n

        # 获取最终的交集
        topk1 = torch.topk(tensor1, C)
        topk2 = torch.topk(tensor2, C)
        # print(f"Final top {ans_k} values for tensor1: {topk1.values},indices:{topk1.indices}")
        # print(f"Final top {ans_k} values for tensor2: {topk2.values},indices:{topk2.indices}")
        indices1 = set(topk1.indices.tolist())
        indices2 = set(topk2.indices.tolist())
        intersection = indices1 | indices2
        # print(tensor1.shape,tensor2.shape)
        # 在交集内，找到 (tensor1 - tensor2) 差值最大的下标
        max_index = max(intersection, key=lambda idx: tensor1[idx] +  alpha*(tensor2[idx] - tensor1[idx]))
        return C, max_index
    def safe_decoding_new3(self,tensor1, tensor2,C,alpha):
        tensor2=tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        topk1 = torch.topk(tensor1, C)
        topk2 = torch.topk(tensor2, C)
        indices1 = topk1.indices.tolist()
        indices2 = topk2.indices.tolist()
        if indices2[0] not in indices1 or indices1[0] not in indices2:
            return C, indices1[0]
        else:
            return self.safe_decoding(tensor1, tensor2,C,alpha)
    def safe_decoding_new4(self,tensor1, tensor2,C,alpha):
        tensor2=tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        left, right = 1, n
        ans_k = None
        while left <= right:
            mid = (left + right) // 2
            indices1 = set(torch.topk(tensor1, mid).indices.tolist())
            indices2 = set(torch.topk(tensor2, mid).indices.tolist())
            if len(indices1 & indices2) >= C:
                ans_k = mid
                right = mid - 1
            else:
                left = mid + 1

        if ans_k is None:
            return None, None  # 若没有满足条件的 k

        # 获取最终的交集
        topk1 = torch.topk(tensor1, ans_k)
        topk2 = torch.topk(tensor2, ans_k)
        indices1 = topk1.indices.tolist()
        indices2 = topk2.indices.tolist()
        if indices2[0] not in indices1 or indices1[0] not in indices2:
            flag=0
            if indices2[0] not in indices1:
                flag=-1
            if indices1[0] not in indices2:
                flag=-2
            if indices2[0] not in indices1 and indices1[0] not in indices2:
                flag=0
            return flag, indices1[0]
        else:   
            indices1 = set(indices1)
            indices2 = set(indices2)
            intersection = indices1 & indices2
            max_index = max(intersection, key=lambda idx: tensor1[idx] +  alpha*(tensor2[idx] - tensor1[idx]))
            return 1, max_index

    def safe_decoding_new5(self,tensor1, tensor2,C,alpha):
        tensor2=tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        left, right = 1, n
        ans_k = None
        while left <= right:
            mid = (left + right) // 2
            indices1 = set(torch.topk(tensor1, mid).indices.tolist())
            indices2 = set(torch.topk(tensor2, mid).indices.tolist())
            if len(indices1 & indices2) >= C:
                ans_k = mid
                right = mid - 1
            else:
                left = mid + 1

        if ans_k is None:
            return None, None  # 若没有满足条件的 k

        # 获取最终的交集
        topk1 = torch.topk(tensor1, ans_k)
        topk2 = torch.topk(tensor2, ans_k)
        indices1 = topk1.indices.tolist()
        indices2 = topk2.indices.tolist()
        if indices1[0] not in indices2:
            flag=-2
            if indices2[0] not in indices1:
                flag=0
            print(f"unmatch:indices2:{indices2[0]}: {self.tokenizer.decode(indices2[0])},indices1:{indices1[0]}: {self.tokenizer.decode(indices1[0])},flag:{flag}")
            if indices2[0] == 306:              
                sorted_indices1 = torch.argsort(tensor1, descending=True)
                rank_306 = (sorted_indices1 == 306).nonzero().item() + 1
                print(f"Token 306 rank in tensor1: {rank_306}")
                print(f"Token 306 value in tensor1: {tensor1[306]}")
                print(f"Top ans_k:{ans_k} values in tensor1: {tensor1[sorted_indices1[:ans_k]]}")
                print(f"Top ans_k:{ans_k} indices in tensor1: {sorted_indices1[:ans_k]}")
        
            return flag, indices1[0]
        else:   
            indices1 = set(indices1)
            indices2 = set(indices2)
            intersection = indices1 & indices2
            max_index = max(intersection, key=lambda idx: tensor1[idx] +  alpha*(tensor2[idx] - tensor1[idx]))
            return 1, max_index
           
    def generate_one_shot_in_batch_by_safeDecoding(self, inputs, accelerator, max_new_tokens = 1024, 
                 do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                 repetition_penalty = 1.0, length_penalty = 1.0,C=5,M=1024,alpha=3.0, **kwargs):
        # a one-shot conversation, input can be a string, a list of messages, or a dictionary with 'messages' key
        # no history will be maintained for one-shot conversation
        # this function is for batch inference to accelerate the evaluation
        
        print(f"max_new_tokens:{max_new_tokens},C:{C},M:{M},alpha:{alpha}")
        function_start_time = time.time()

        M = max_new_tokens
        inputs_processed = []

        for item in inputs:

            if isinstance(item, dict) or isinstance(item, list):
                item_processed = self.validate_conversation(item)
            elif isinstance(item, str):
                item_processed = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
            else:
                raise ValueError(f"input {item} is not a valid conversation input")
            
            item_processed = self.string_formatter({'messages': item_processed})['text']

            inputs_processed.append(item_processed)
        

        model_inputs = self.tokenizer(inputs_processed, padding = True, return_tensors="pt").to(self.model.device)
        # 初始化生成的输入（假设model_inputs是初始输入）
        generated_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        warning_flag1 = False

        # wxk
        for step in range(M):
            start_time = time.time()
            # 模型前向传播
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            expert_outputs = self.expert_model(
                input_ids=generated_ids.to(self.expert_model.device),
                attention_mask=attention_mask.to(self.expert_model.device),
                return_dict=True
            )

            logits = outputs.logits
            expert_logits = expert_outputs.logits
            # 获取最后一个token的logits
            last_token_logits = logits[0, -1, :]
            expert_last_token_logits =  expert_logits[0, -1, :]

            if(last_token_logits.shape != expert_last_token_logits.shape and not warning_flag1):
                print(f"warning:last_token_logits.shape is {last_token_logits.shape},while expert_last_token_logits.shape is {expert_last_token_logits.shape}")
                warning_flag1=True
            # # 应用logits处理器（温度、top-p、重复惩罚等）
            # for processor in logits_processor:
            #     last_token_logits = processor(generated_ids, last_token_logits)

            # 转换为概率
            probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
            expert_probabilities = torch.nn.functional.softmax(expert_last_token_logits, dim=-1)

            get_probabilities_time = time.time()
            print(f"get_probabilities_time: {get_probabilities_time - start_time}")
            # 执行算法，贪婪采样
            k,next_token = self.safe_decoding_new(tensor1 = probabilities,tensor2 = expert_probabilities,C=C,alpha=alpha)
            get_next_token_time = time.time()
            print(f"get_next_token_time: {get_next_token_time - get_probabilities_time}")   
            next_token = torch.Tensor([next_token]).to(generated_ids.device)
            
            # 终止条件：遇到EOS或达到最大长度
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # 更新生成序列和attention mask
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).long()], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=attention_mask.device)], dim=1)
            end_time = time.time()
            print(f"step {step} total time: {end_time - start_time}")
        

        # print(max_new_tokens-M)
        if max_new_tokens-M > 0:
            generated_ids = self.model.generate(
                input_ids = generated_ids,
                attention_mask = attention_mask,
                max_new_tokens=(max_new_tokens-M),
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                stopping_criteria = self.stopping_criteria,
                **kwargs
            )
        generation_end_time = time.time()
        print(f"generation time: {generation_end_time - function_start_time}")
        # print(generated_ids.shape,outputs.shape)
        full_texts = [] # the whole conversation texts
        output_texts = [] # the model output part texts

        for i, item in enumerate(generated_ids):

            input_pos = model_inputs['attention_mask'][i].nonzero()

            input_length = input_pos.shape[0] # how many input tokens
            start_pos = input_pos[0][0] # the first non-padding token

            full_text = self.tokenizer.decode(item, skip_special_tokens=True)
            output_text = self.tokenizer.decode(item[start_pos + input_length:], skip_special_tokens=True)

            full_texts.append(full_text)
            output_texts.append(output_text)

        function_end_time = time.time()
        print(f"function time: {function_end_time - function_start_time}")
        return output_texts, full_texts
    
    def generate_one_shot_in_batch_by_safeDecoding_with_speculative_decoding(self, inputs, accelerator, max_new_tokens = 1024, 
                do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                repetition_penalty = 1.0, length_penalty = 1.0,C=10,M=1024,alpha=0.5, **kwargs):

   
        M = max_new_tokens
        print(f"max_new_tokens:{max_new_tokens},C:{C},M:{M},alpha:{alpha}")
        inputs_processed = []
        # function_start_time = time.time()
        for item in inputs:
            if isinstance(item, dict) or isinstance(item, list):
                item_processed = self.validate_conversation(item)
            elif isinstance(item, str):
                item_processed = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
            else:
                raise ValueError(f"input {item} is not a valid conversation input")
            
            item_processed = self.string_formatter({'messages': item_processed})['text']

            inputs_processed.append(item_processed)
        
        # print(f"inputs_processed:{inputs_processed}")

        model_inputs = self.tokenizer(inputs_processed, padding = True, return_tensors="pt").to(self.model.device)
        generated_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        
        num_speculate_tokens = 7
        threshold = 0.5
        warning_flag1 = False

        with torch.no_grad():
            # 初始化生成过程
            cur_len=0
            # extra_match_tokens=0
            is_token_match_log=[]
            is_safe_decoding_log=[]
            while cur_len < M:
                # start_time = time.time()
                expert_probability_distributions=[]
                expert_speculate_ids = []
                raw_output = generated_ids.to(self.expert_model.device)
                prepared_attention_masks = [attention_mask]
                for i in range(num_speculate_tokens):
                    prepared_attention_masks.append(torch.cat([prepared_attention_masks[i],torch.ones((1, 1), device=attention_mask.device)],dim=-1))

                # 猜测的token
                for _ in range(num_speculate_tokens):
                    # 获取当前 token 的 logits
                    outputs =self.expert_model(
                    input_ids=raw_output.to(self.expert_model.device),
                    attention_mask=prepared_attention_masks[_],
                    return_dict=True
                )
                    logits = outputs.logits
                    
                    # 获取当前最后一个 token 的 logits（假设是 batch_size=1）
                    last_token_logits = logits[0, -1, :]
                    
                    # 将 logits 转换为概率分布
                    probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
                    
                    # 保存当前的概率分布
                    expert_probability_distributions.append(probabilities)
                    
                    # 贪婪选择最大概率的 token
                    next_token = torch.argmax(probabilities, dim=-1)
                    # print(f"next_token,{next_token.item(),next_token.shape}")
                    # print(raw_output.shape)
                    expert_speculate_ids.append(next_token.item())
                    
                    # 将选中的 token 添加到生成序列中(不需要最后一个)
                    if _ != num_speculate_tokens-1:
                        raw_output = torch.cat((raw_output, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                    # generated_expert_text = self.tokenizer.decode(expert_speculate_ids, skip_special_tokens=True)
                    # print("\nGenerated expert text:", generated_expert_text)
                
                # get_speculate_time = time.time()
                # print(f"get_speculate_time: {get_speculate_time - start_time}")

                outputs = self.model(
                    input_ids=raw_output,
                    attention_mask=prepared_attention_masks[-2],
                    return_dict=True
                )
                logits = outputs.logits
                last_token_logits = logits[0, -num_speculate_tokens:, :]
                
                # get_logits_time = time.time()
                # print(f"get_logits_time: {get_logits_time - get_speculate_time}")

                # 转换为概率
                probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
                cur_tokens=[]
                break_flag = False
                for i in range(num_speculate_tokens):
                    # 执行算法，贪婪采样
                    correct_flag = False
                    k,cur_token = self.safe_decoding_new5(tensor1 = probabilities[i-num_speculate_tokens,:],tensor2 =expert_probability_distributions[i],C=C,alpha=alpha)
                    is_safe_decoding_log.append(k)
                    if(cur_token == expert_speculate_ids[i]):
                        correct_flag = True # 可以检查下一步
                        is_token_match_log.append(1) # 这一步match了
                    else:
                        is_token_match_log.append(0)
#---------------------------------------------------------new6
                    # if len(is_token_match_log) % num_speculate_tokens == 0:
                    #     if(sum(is_token_match_log[-num_speculate_tokens:])/ num_speculate_tokens < threshold):
                    #         alpha = 1.0
                    #     else:
                    #         alpha = 0.5
                    #     print(f"cur_len:{cur_len},alpha:{alpha}")
#---------------------------------------------------------
#---------------------------------------------------------new7
                    if len(is_token_match_log) % num_speculate_tokens == 0:
                        if(sum(is_token_match_log[-num_speculate_tokens:])/ num_speculate_tokens < threshold):
                            next_alpha = 1.0
                        else:
                            next_alpha = 0.5
                        if next_alpha == alpha:
                            threshold -= max_new_tokens//num_speculate_tokens * 0.5
                        else:
                            threshold = 0.5
                        alpha = next_alpha
                        print(f"cur_len:{cur_len},alpha:{alpha},threshold:{threshold}")
#---------------------------------------------------------                        
                    cur_token = torch.Tensor([cur_token]).unsqueeze(0).long().to(generated_ids.device)   
                    cur_tokens.append(cur_token) 
                    cur_len += 1

                    if(cur_len >= M):
                        break_flag = True
                        break
                    # 终止条件：遇到EOS或达到最大长度
                    if cur_token.item() == self.tokenizer.eos_token_id:
                        break_flag = True
                        break
                    if correct_flag == False:
                        break
                # extra_match_tokens +=  len(cur_tokens)-1
                generated_ids = torch.cat([generated_ids]+cur_tokens, dim=-1)
                attention_mask = prepared_attention_masks[len(cur_tokens)]

                # print("extra_match_tokens:",extra_match_tokens)

                # validating_end_time = time.time()
                # print("validating time:",validating_end_time - get_speculate_time)
                # print(f"step {cur_len} total epoch time: {validating_end_time - start_time},extra match tokens:{extra_match_tokens}")
                if break_flag:
                    break
        # generation_end_time = time.time()
        # print(f"generation time: {generation_end_time - function_start_time},time per token:{(generation_end_time - function_start_time)/cur_len}")  
            # Split the list into groups of `group_size`
        groups = [is_token_match_log[i:i + num_speculate_tokens] for i in range(0, len(is_token_match_log), num_speculate_tokens)]
        
        # Calculate the ratio of 1's in each group
        ones_ratio = [sum(group) / len(group) for group in groups]
        print("speculative matching :",ones_ratio)
        print("safe decoding:",is_safe_decoding_log)
        # print(generated_ids.shape,attention_mask.shape)
        if max_new_tokens-M > 0:
            generated_ids = self.model.generate(
                input_ids = generated_ids,
                attention_mask = attention_mask,
                max_new_tokens=(max_new_tokens-M),
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                stopping_criteria = self.stopping_criteria,
                **kwargs
            )
        full_texts = [] # the whole conversation texts
        output_texts = [] # the model output part texts

        for i, item in enumerate(generated_ids):

            input_pos = model_inputs['attention_mask'][i].nonzero()

            input_length = input_pos.shape[0] # how many input tokens
            start_pos = input_pos[0][0] # the first non-padding token

            full_text = self.tokenizer.decode(item, skip_special_tokens=True)
            output_text = self.tokenizer.decode(item[start_pos + input_length:], skip_special_tokens=True)

            full_texts.append(full_text)
            output_texts.append(output_text)

        # function_end_time = time.time()
        # print(f"function time: {function_end_time - function_start_time}")
        return output_texts, full_texts
