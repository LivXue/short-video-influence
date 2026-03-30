import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from copy import deepcopy

import torch
from torch.nn import CrossEntropyLoss
from deepspeed.pipe import PipelineModule, LayerSpec

from transformers import NetQwen2VL, NetQwen2vlProcessor, NetQwen2VL_stage1, NetQwen2VL_stage2, NetQwen2VL_stage3, NetQwen2VL_stage4, NetQwen2VL_Reg


class Loss(torch.nn.Module):
   def __init__(self, config):
        super().__init__()
        self.loss_fct = CrossEntropyLoss()
        self.config = config

   def forward(self, logits, labels):
      # Shift so that tokens < n predict n
      shift_logits = logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      shift_logits = shift_logits.view(-1, self.config.vocab_size)
      shift_labels = shift_labels.view(-1)
      # Enable model parallelism
      shift_labels = shift_labels.to(shift_logits.device)
      loss = self.loss_fct(shift_logits, shift_labels)

      return loss

class MultiStagePipeline(PipelineModule):
    def __init__(self, model):
        specs = [
            LayerSpec(NetQwen2VL_stage1, model),
            LayerSpec(NetQwen2VL_stage2, model),
            LayerSpec(NetQwen2VL_stage3, model),
            LayerSpec(NetQwen2VL_stage4, model)
        ]
        super().__init__(specs, num_stages=4, loss_fn=Loss(model.config))
        self.complete_model = model.to('cpu')

    def get_model(self):
        layer_list = self.forward_funcs
        layer_list = [_.to('cpu') for _ in layer_list]
        stage1, stage2, stage3, stage4 = layer_list
        self.complete_model.graph_projector = stage1.graph_projector
        self.complete_model.model.embed_tokens = stage1.embed_tokens
        self.complete_model.lm_head = stage4.lm_head

        return self.complete_model

def initial_model(device="balanced"):
   chat_template = json.load(open("NetQwen2VL_chat_template.json"))["chat_template"]
   processor = NetQwen2vlProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir="./NetQwen2-VL-7B-Instruct")
   processor.tokenizer.chat_template = chat_template
   processor.chat_template = chat_template
   graph_token_id = processor.tokenizer.convert_tokens_to_ids(['<|graph_pad|>'])[0]
   graph_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<|graph_start|>'])[0]
   graph_end_token_id = processor.tokenizer.convert_tokens_to_ids(['<|graph_end|>'])[0]
   model = NetQwen2VL.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", cache_dir="./NetQwen2-VL-7B-Instruct")#, device_map=device)
   model.config.graph_token_id = graph_token_id
   model.config.graph_start_token_id = graph_start_token_id
   model.config.graph_end_token_id = graph_end_token_id

   model.freeze_parameters()

   return model, processor


def initial_model2(device="balanced"):
   processor = NetQwen2vlProcessor.from_pretrained("./NetQwen2-VL-7B-sft")
   model = NetQwen2VL_Reg.from_pretrained("./NetQwen2-VL-7B-sft", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")#, device_map=device)
   chat_template = json.load(open("NetQwen2VL_chat_template.json"))["chat_template"]
#    processor = NetQwen2vlProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir="./NetQwen2-VL-7B-Instruct")
   processor.tokenizer.chat_template = chat_template
   processor.chat_template = chat_template
#    model = NetQwen2VL_Reg.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", cache_dir="./NetQwen2-VL-7B-Instruct")
   graph_token_id = processor.tokenizer.convert_tokens_to_ids(['<|graph_pad|>'])[0]
   graph_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<|graph_start|>'])[0]
   graph_end_token_id = processor.tokenizer.convert_tokens_to_ids(['<|graph_end|>'])[0]
   model.config.graph_token_id = graph_token_id
   model.config.graph_start_token_id = graph_start_token_id
   model.config.graph_end_token_id = graph_end_token_id

   model.freeze_parameters()

   return model, processor


def divided_model():
    model, processor = initial_model()

    return MultiStagePipeline(model), processor


if __name__ == "__main__":
    model, processor = initial_model()
