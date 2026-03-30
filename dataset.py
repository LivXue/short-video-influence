import json
import os
from copy import deepcopy

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import pickle

from qwen_vl_utils import process_vision_info


class NetGPT_dataset(Dataset):
    def __init__(self, json_file, processor):
        super().__init__()
        self.data = json.load(open(json_file))
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = 5000

    def __getitem__(self, index):
        return self.process_func(self.data[index])
    
    def process_func(self, example):
        """
        将数据集进行预处理
        """
        instruction = [example[0]]
        text = self.processor.apply_chat_template(
            instruction, tokenize=False, add_generation_prompt=True
        )  # 获取文本
        graph_inputs, video_inputs = process_vision_info(example)  # 获取数据数据（预处理过）
        inputs = self.processor(
            text=[text],
            graphs=graph_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        inputs = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

        instruction = inputs
        response = self.tokenizer(example[1]["content"][0]["text"], add_special_tokens=False)

        input_ids = (
                instruction["input_ids"][0] + response["input_ids"] + [self.tokenizer.pad_token_id]
        )

        attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
        labels = (
                [-100] * len(instruction["input_ids"][0])
                + response["input_ids"]
                + [self.tokenizer.pad_token_id]
        )
        # if len(input_ids) > MAX_LENGTH:  # 做一个截断
        #     input_ids = input_ids[:MAX_LENGTH]
        #     attention_mask = attention_mask[:MAX_LENGTH]
        #     labels = labels[:MAX_LENGTH]

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        graph_feature = inputs['graph_feature'][0].type(torch.bfloat16)
        pixel_values_videos = torch.tensor(inputs['pixel_values_videos'], dtype=torch.bfloat16)
        video_grid_thw = torch.tensor(inputs['video_grid_thw'][0], dtype=torch.int)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "graph_feature": graph_feature,
                "pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw}
        #return [graph_feature, input_ids, attention_mask, labels, pixel_values_videos, video_grid_thw]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        max_length = self.max_length#max([len(data["input_ids"]) for data in batch])
        for data in batch:
            add_length = max_length - len(data["input_ids"])
            if add_length <= 0:
                data["input_ids"] = torch.cat([data["input_ids"][:max_length-5], data["input_ids"][-5:]])
                data["labels"] = torch.cat([data["labels"][:max_length-5], data["labels"][-5:]])
                data["attention_mask"] = torch.cat([data["attention_mask"][:max_length-5], data["attention_mask"][-5:]])
            else:
                data["input_ids"] = torch.cat([data["input_ids"], torch.full([add_length], self.tokenizer.pad_token_id, dtype=data["input_ids"].dtype)])
                data["labels"] = torch.cat([data["labels"], torch.full([add_length], self.tokenizer.pad_token_id, dtype=data["labels"].dtype)])
                data["attention_mask"] = torch.cat([data["attention_mask"], torch.zeros([add_length], dtype=data["attention_mask"].dtype)])
        
        keys = batch[0].keys()
        collected_batch = {k: torch.stack([data[k] for data in batch]).contiguous() for k in keys}
        return ((collected_batch["graph_feature"], collected_batch["input_ids"], collected_batch["attention_mask"], collected_batch["labels"], collected_batch["pixel_values_videos"], collected_batch["video_grid_thw"]), collected_batch["labels"])
    

class NetGPT_fast_dataset(Dataset):
    def __init__(self, json_file, processor):
        super().__init__()
        self.data = json.load(open(json_file))
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self.max_length = 4000

        data_path = f"data/precomputed_data_{json_file.split('.')[0]}.pkl"
        if os.path.exists(data_path):
            self.input_ids, self.attention_mask, self.labels, self.graph_feature = torch.load(data_path)
        else:
            self.preprocess_data()
            torch.save([self.input_ids, self.attention_mask, self.labels, self.graph_feature], data_path)
        
    def preprocess_data(self):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.graph_feature = []

        for datum in tqdm(self.data):
            input_ids, attention_mask, labels, graph_feature = self.process_func(datum)
            if len(input_ids) > self.max_length:
                continue
            self.input_ids.append(input_ids)
            self.attention_mask.append(attention_mask)
            self.labels.append(labels)
            self.graph_feature.append(graph_feature)


    def __getitem__(self, index):
        input_ids, attention_mask, labels, graph_feature = \
            self.input_ids[index], self.attention_mask[index], self.labels[index], self.graph_feature[index]
        
        # Pad token length
        add_length = self.max_length - len(input_ids)
        input_ids = torch.cat([input_ids, torch.full([add_length], self.tokenizer.pad_token_id, dtype=input_ids.dtype)])
        labels = torch.cat([labels, torch.full([add_length], -100, dtype=labels.dtype)])
        attention_mask = torch.cat([attention_mask, torch.zeros([add_length], dtype=attention_mask.dtype)])

        return input_ids, attention_mask, labels, graph_feature
    
    def process_func(self, example):
        """
        将数据集进行预处理
        """
        instruction = [example[0]]
        text = self.processor.apply_chat_template(
            instruction, tokenize=False, add_generation_prompt=True
        )  # 获取文本
        graph_inputs, _ = process_vision_info(example)  # 获取数据数据（预处理过）
        inputs = self.processor(
            text=[text],
            graphs=graph_inputs,
            padding=False,
            return_tensors="pt",
        )
        inputs = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

        instruction = inputs
        response = self.tokenizer(example[1]["content"][0]["text"], add_special_tokens=False)

        input_ids = (
                instruction["input_ids"][0] + response["input_ids"] + [self.tokenizer.pad_token_id]
        )

        attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
        labels = (
                [-100] * len(instruction["input_ids"][0])
                + response["input_ids"]
                + [self.tokenizer.pad_token_id]
        )

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        graph_feature = inputs['graph_feature'][0].type(torch.bfloat16)
        return input_ids, attention_mask, labels, graph_feature

    def __len__(self):
        return len(self.input_ids)

    def collate_fn(self, batch):
        collected_batch = {}
        collected_batch["input_ids"] = torch.stack([sample[0] for sample in batch])
        collected_batch["attention_mask"] = torch.stack([sample[1] for sample in batch])
        collected_batch["labels"] = torch.stack([sample[2] for sample in batch])
        collected_batch["graph_feature"] = torch.stack([sample[3] for sample in batch])
        return collected_batch
    

class NetGPT_fast_dataset2(Dataset):
    def __init__(self, json_file, processor):
        super().__init__()
        self.data = json.load(open(json_file))
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self.max_length = 4000

        data_path = f"data/precomputed_data_{json_file.split('.')[0]}_reg.pkl"
        if os.path.exists(data_path):
            self.input_ids, self.attention_mask, self.labels, self.graph_feature = torch.load(data_path)
        else:
            self.preprocess_data()
            torch.save([self.input_ids, self.attention_mask, self.labels, self.graph_feature], data_path)
        
    def preprocess_data(self):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.graph_feature = []

        for datum in tqdm(self.data):
            input_ids, attention_mask, labels, graph_feature = self.process_func(datum)
            if len(input_ids) > self.max_length:
                continue
            self.input_ids.append(input_ids)
            self.attention_mask.append(attention_mask)
            self.labels.append(labels)
            self.graph_feature.append(graph_feature)


    def __getitem__(self, index):
        input_ids, attention_mask, labels, graph_feature = \
            self.input_ids[index], self.attention_mask[index], self.labels[index], self.graph_feature[index]
        
        # Pad token length
        add_length = self.max_length - len(input_ids)
        input_ids = torch.cat([input_ids, torch.full([add_length], self.tokenizer.pad_token_id, dtype=input_ids.dtype)])
        attention_mask = torch.cat([attention_mask, torch.zeros([add_length], dtype=attention_mask.dtype)])

        return input_ids, attention_mask, labels, graph_feature
    
    def process_func(self, example):
        """
        将数据集进行预处理
        """
        instruction = [example[0]]
        text = self.processor.apply_chat_template(
            instruction, tokenize=False, add_generation_prompt=True
        )  # 获取文本
        graph_inputs, _ = process_vision_info(example)  # 获取数据数据（预处理过）
        inputs = self.processor(
            text=[text],
            graphs=graph_inputs,
            padding=False,
            return_tensors="pt",
        )
        inputs = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

        instruction = inputs

        input_ids = (
                instruction["input_ids"][0]
        )

        attention_mask = instruction["attention_mask"][0]
        labels = float(example[1]["content"][0]["text"])

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        graph_feature = inputs['graph_feature'][0]#.type(torch.bfloat16)
        return input_ids, attention_mask, labels, graph_feature

    def __len__(self):
        return len(self.input_ids)

    def collate_fn(self, batch):
        collected_batch = {}
        collected_batch["input_ids"] = torch.stack([sample[0] for sample in batch])
        collected_batch["attention_mask"] = torch.stack([sample[1] for sample in batch])
        collected_batch["labels"] = torch.stack([sample[2] for sample in batch])
        collected_batch["graph_feature"] = torch.stack([sample[3] for sample in batch])
        return collected_batch
    

if __name__ == "__main__":
    from initial_NetLLM import initial_model
    _, processor = initial_model()
    train_set = NetGPT_fast_dataset2("netqwen_train2.json", processor)
    test_set = NetGPT_fast_dataset2("netqwen_test2.json", processor)
