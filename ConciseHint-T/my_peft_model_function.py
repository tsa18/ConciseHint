from __future__ import annotations
import collections
import copy
import inspect
import os
import warnings
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory, named_module_tensors
from huggingface_hub import HfFileSystem, ModelCard, ModelCardData, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils.constants import DUMMY_MODEL_CONFIG
from peft.utils.integrations import init_empty_weights
from peft.utils.other import TrainableTokensWrapper
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftType,
    TaskType,
    _get_batch_size,
    _get_input_embeddings_name,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    id_tensor_storage,
    infer_device,
    load_peft_weights,
    map_cache_to_layer_device_map,
    set_peft_model_state_dict,
    shift_tokens_right,
)
import torch

def my_peft_model_for_causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    task_ids=None,
    length_level = None,
    **kwargs,
):
    # print('my forward function')
    peft_config = self.active_peft_config
    if not peft_config.is_prompt_learning:
        if self.base_model.config.model_type == "mpt":
            if inputs_embeds is not None:
                raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        if peft_config.peft_type == PeftType.POLY:
            kwargs["task_ids"] = task_ids

        with self._enable_peft_forward_hooks(**kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

    batch_size = _get_batch_size(input_ids, inputs_embeds)
    if attention_mask is not None:
        prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    if kwargs.get("position_ids", None) is not None:
        warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        kwargs["position_ids"] = None
    if kwargs.get("token_type_ids", None) is not None:
        warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
        kwargs["token_type_ids"] = None
    kwargs.update(
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
    )

    if peft_config.peft_type == PeftType.PREFIX_TUNING:
        # overwrite past_kv in kwargs
        kwargs["past_key_values"] = self.get_prompt(batch_size)
        return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
    elif peft_config.peft_type == PeftType.CPT:
        return self._cpt_forward(input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs)
    else:
        
        if hasattr(peft_config, "use_concise_tuning") and peft_config.use_concise_tuning:


            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # obtain prompt embeddings
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)

            interval = [64,96,128,160,192][length_level] # 0-4

            from transformers import AutoTokenizer
            import numpy
            tokenizer = AutoTokenizer.from_pretrained(self.base_model.config.name_or_path)
            think_id = tokenizer.encode('<think>')[0]
            end_think_id = tokenizer.encode('</think>')[0]
            think_index = input_ids[0].cpu().numpy().tolist().index(think_id)

            # adjust input
            inputs_embeds_before_think = inputs_embeds[:, 0:think_index+1]
            inputs_embeds_after_think = inputs_embeds[:, think_index+1:]

            # split input
            chunks = torch.split(inputs_embeds_after_think, interval, dim=1)
            chunks_input_ids =  torch.split( input_ids[:, think_index+1:], split_size_or_sections = interval , dim=1)

            ### construct new input, insert hint embeddings
            combined_embeds = []
            for i, (chunk, chunk_input_ids) in enumerate(zip(chunks, chunks_input_ids )):
                assert chunk.shape[0] ==1 
                combined_embeds.append(prompts)
                combined_embeds.append(chunk)
            new_inputs_embeds = torch.cat( [inputs_embeds_before_think] + combined_embeds, dim=1)

            
            # adjust labels（set labels of hint to -100）
            if labels is not None:

                labels_before_think = labels[:, 0:think_index+1]
                labels_after_think = labels[:, think_index+1:]
                insert_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                # split original labels
                label_chunks = torch.split( labels_after_think, interval, dim=1)
                combined_labels = []
                for i, (chunk, chunk_input_ids) in enumerate(zip(label_chunks, chunks_input_ids)):
                    combined_labels.append(insert_labels)
                    combined_labels.append(chunk)
                kwargs["labels"] = torch.cat( [labels_before_think] + combined_labels, dim=1)
        

            # adjust attention mask 
            assert new_inputs_embeds.shape[0] == 1
            kwargs['attention_mask'] = torch.ones(new_inputs_embeds.shape[0],new_inputs_embeds.shape[1])
            
            return self.base_model(inputs_embeds=new_inputs_embeds, **kwargs)

        else:
            ### ori
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)



def my_prepare_inputs_for_generation(self, *args, task_ids= None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.38, all architectures should support caching.
        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        if packaging.version.parse(transformers.__version__) > packaging.version.parse("4.43.3"):
            # https://github.com/huggingface/transformers/pull/31445
            transformers_new_cache_archs.append("bloom")

        uses_cache = uses_transformers_4_38 or (
            uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs.get("past_key_values", None) is not None):
                # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
                # In prompt learning methods, past key values are longer when compared to the `input_ids`.
                # As such only consider the last input ids in the autogressive generation phase.
                past_key_values = model_kwargs["past_key_values"]
                if isinstance(past_key_values, (tuple, list)):
                    seq_len = past_key_values[0][0].shape[-2]
                else:  # using transformers kv cache
                    seq_len = past_key_values.get_seq_length()
                if seq_len >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

            if model_kwargs.get("attention_mask", None) is not None:
                pass

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                # prepare position ids
                if hasattr(peft_config, "inserted_prompt_num"):
                    inserted_prompt_num = peft_config.inserted_prompt_num
                    model_kwargs["position_ids"] = model_kwargs["position_ids"] + peft_config.num_virtual_tokens*inserted_prompt_num

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None


            # no past_key_values or past_key_values empty cache
            requires_prompt_injection = (model_kwargs.get("past_key_values", None) is None) or (
                isinstance(model_kwargs["past_key_values"], transformers.Cache)
                and not model_kwargs["past_key_values"].get_seq_length()
            )

            # add extra attention mask brought by prompt injection
            if hasattr(peft_config,"inserted_prompt_num"):
                inserted_prompt_num = peft_config.inserted_prompt_num
                size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens*inserted_prompt_num
                extra_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat( (model_kwargs["attention_mask"], extra_attention_mask), dim=1 )              
            

            if requires_prompt_injection and peft_config.peft_type == PeftType.PREFIX_TUNING:
                new_past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = new_past_key_values
            elif requires_prompt_injection:
                if hasattr(peft_config, "use_concise_tuning") and peft_config.use_concise_tuning:

                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.base_model.config.name_or_path, padding_side="left")
                    intervals = peft_config.intervals
                    print('inference concise, intervals:', intervals)
                    input_ids = model_kwargs["input_ids"]
                    inputs_embeds = self.word_embeddings(input_ids)


                    # obtain prompt embeddings
                    prompts = self.get_prompt(batch_size=input_ids.shape[0], task_ids=task_ids)
                    prompts = prompts.to(inputs_embeds.dtype)

                    #### use no finetuning
                    prompts_manual = ["-<|im_start|>user\nmake answer concise!<|im_end|>-"] * input_ids.shape[0]
                    prompts_ids_manual = tokenizer(prompts_manual, add_special_tokens=False).input_ids
                    prompts_manual = self.word_embeddings(torch.tensor(prompts_ids_manual).to(input_ids))
                    alpha = 1.0 # ours
                    # alpha = 0.0 # non-finetune
                    # alpha = 0.6
                    prompts = alpha*prompts + (1-alpha)*prompts_manual



                    # # locate think
                    think_id = tokenizer.encode('<think>')[0]
                    import numpy
                    think_index = input_ids[0].cpu().numpy().tolist().index(think_id)  
                    inputs_embeds_before_think = inputs_embeds[:, 0:think_index+1]
                    inputs_embeds_after_think = inputs_embeds[:, think_index+1:]
                    # split
                    chunks = torch.split(inputs_embeds_after_think, split_size_or_sections = intervals[:-1] , dim=1)
                    chunks_input_ids =  torch.split( input_ids[:, think_index+1:], split_size_or_sections = intervals[:-1] , dim=1)
                    
        
                    # construct new input
                    combined_embeds = []
                    inserted_prompt_num = 0
                    for i, (chunk, chunk_input_ids) in enumerate(zip(chunks, chunks_input_ids)):
                        if chunk.shape[1]==0: continue

                        n = chunk.shape[1]
                        start_ratio = min( max(n-128, 0)/1024, 0.8)
                        k =  int(start_ratio* (n-1))
                        combined_embeds.append(chunk[:, 0:k])
                        combined_embeds.append(prompts)
                        inserted_prompt_num +=1
                        combined_embeds.append(chunk[:, k:])
                        

                    setattr(peft_config,"inserted_prompt_num", inserted_prompt_num)
                    new_inputs_embeds = torch.cat( [inputs_embeds_before_think] + combined_embeds, dim=1)
                    model_kwargs["inputs_embeds"] = new_inputs_embeds # concise
                    model_kwargs["input_ids"] = None

                    def generate_custom_left_padding_attention_mask(input_ids: torch.Tensor, pad_token_id: int, eos_token_id: int) -> torch.Tensor:
                        """
                        生成自定义的attention_mask，遵循以下规则：
                        1. 默认情况下，pad_token_id 和 eos_token_id 对应的mask为0。
                        2. 从左到右扫描，一旦遇到第一个既非pad_token_id也非eos_token_id的token，
                        该token及其之后的所有token（包括后续的pad_token_id和eos_token_id）的mask都置为1。

                        Args:
                            input_ids (torch.Tensor): 输入的token ID张量，形状为 (batch_size, sequence_length)。
                            pad_token_id (int): 填充token的ID。
                            eos_token_id (int): 序列结束token的ID。

                        Returns:
                            torch.Tensor: 生成的attention mask张量，形状与input_ids相同，dtype为torch.int。
                        """
                        batch_size, seq_len = input_ids.shape

                        # 1. 识别“内容”token（即不是pad也不是eos的token）
                        # 这会生成一个布尔张量，形状与 input_ids 相同
                        # 例如：[0, 0, 0, 1, 1, 1, 0] (A,B,C是内容，PAD是0)
                        content_indicators = (input_ids != pad_token_id) & (input_ids != eos_token_id)

                        # 2. 找到每个序列中第一个“内容”token的索引
                        # torch.argmax 在布尔张量上工作时，会返回第一个 True (1) 的索引。
                        # 如果一行全是 False (即没有内容token，全是pad/eos)，argmax会返回0。
                        first_content_indices = torch.argmax(content_indicators.int(), dim=1) # shape: (batch_size,)

                        # 3. 处理没有“内容”token的序列
                        # 对于那些完全由pad/eos组成的序列（`content_indicators.any(dim=1)` 为 False），
                        # 它们的 `first_content_indices` 仍然是 0。我们希望它们的mask全是0。
                        # 解决方法是将其 `first_content_indices` 设置为 `seq_len`，这样后续的 `>=` 比较就会全部为 False。
                        has_content = content_indicators.any(dim=1) # shape: (batch_size,)
                        first_content_indices[~has_content] = seq_len

                        # 4. 基于 `first_content_indices` 构建最终的 attention mask
                        # 创建一个与序列长度相同的索引范围张量，形状为 (1, seq_len)
                        col_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

                        # 将 first_content_indices 扩展为 (batch_size, 1)，以便与 col_indices 进行广播比较
                        first_content_indices_expanded = first_content_indices.unsqueeze(1)

                        # 比较：如果列索引 (j) 大于等于该行第一个内容token的索引 (i)，则为1，否则为0
                        # 结果张量形状为 (batch_size, seq_len)
                        attention_mask = (col_indices >= first_content_indices_expanded).int()
                        
                        return attention_mask

                    def generate_position_ids_with_mask(
                        attention_mask: torch.Tensor
                    ) -> torch.Tensor:
                     
                        batch_size, sequence_length = attention_mask.shape
                        device = attention_mask.device

                        # 1. 初始化 position_ids 张量，所有位置都设为 1。
                        #    对于 attention_mask 为 0 的位置，我们希望 position_id 保持为 1。
                        position_ids = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)

                        # 2. 计算每个位置之前有多少个 *不活跃* (attention_mask == 0) 的 token。
                        #    这将作为我们计算实际 position_id 的偏移量。
                        #    示例: attention_mask = [0, 0, 0, 1, 1, 1]
                        #    (attention_mask == 0).long() -> [1, 1, 1, 0, 0, 0]
                        #    torch.cumsum(...)          -> [1, 2, 3, 3, 3, 3]
                        #    这个 `cumulative_leading_zeros`  tensor 包含了每个位置前的“无效”token数量。
                        #    例如，在索引 3 (即第四个位置)时，其值为 3，表示前面有 3 个无效 token。
                        cumulative_leading_zeros = torch.cumsum((attention_mask == 0).long(), dim=1)

                        # 3. 创建一个基础的 position_id 序列：[0, 1, 2, ..., sequence_length - 1]
                        #    并将其扩展到批次维度。
                        base_indices = torch.arange(sequence_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

                        # 4. 计算“活跃”位置的 position_ids。
                        #    对于每个活跃 token (attention_mask 为 1)，它的 position_id 应该是
                        #    `其在整个序列中的索引` 减去 `其前面有多少个无效 token`。
                        #    示例: base_indices = [0,1,2,3,4,5], cumulative_leading_zeros = [1,2,3,3,3,3]
                        #    proposed_active_positions = [0-1, 1-2, 2-3, 3-3, 4-3, 5-3]
                        #                              = [-1, -1, -1, 0, 1, 2]
                        proposed_active_positions = base_indices - cumulative_leading_zeros

                        # 5. 使用 torch.where 将正确计算的 position_ids 应用到 `position_ids` 张量。
                        #    如果 attention_mask 为 1 (活跃 token)，则使用 proposed_active_positions 的值。
                        #    如果 attention_mask 为 0 (非活跃/填充 token)，则使用初始值 1。
                        position_ids = torch.where(
                            attention_mask == 1,
                            proposed_active_positions,
                            position_ids  # 保持为1
                        )

                        return position_ids


                    attention_mask = generate_custom_left_padding_attention_mask(input_ids[:, 0:think_index+1], tokenizer.pad_token_id, tokenizer.eos_token_id)
                    attention_mask = torch.cat( [attention_mask, torch.ones(input_ids.shape[0],  model_kwargs["inputs_embeds"].shape[1]-(think_index+1) ).to(attention_mask)], dim=1 ) 
                    model_kwargs["attention_mask"] = attention_mask
                    model_kwargs["position_ids"] = generate_position_ids_with_mask(model_kwargs["attention_mask"])

                    # # # ori, no prompt injection
                    # model_kwargs["inputs_embeds"] = inputs_embeds
                    # model_kwargs["input_ids"] = None
                 


                else: # ori prompt tuning 
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None


        # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
        # passed in the forward pass to keep track of the position ids of the cache. We have to
        # pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
        # `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
        _ = model_kwargs.pop("cache_position", None)
        return model_kwargs