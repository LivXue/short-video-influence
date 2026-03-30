import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed.utils import RepeatingLoader
#from accelerate import Accelerator

from initial_NetLLM import initial_model2, divided_model
from metrics import *
from dataset import *


if __name__ == "__main__":
    #arguments
    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    save_dir = "NetQwen2-VL-7B-Pipeline"

    # initialization
    deepspeed.init_distributed()

    model, processor = initial_model2()
    train_dataset = NetGPT_fast_dataset2("netqwen_train2.json", processor)
    test_dataset = NetGPT_fast_dataset2("netqwen_test2.json", processor)
    #train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=test_dataset.collate_fn)
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    #optimizer = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    model_engine, _, train_dataloader, _ \
        = deepspeed.initialize(args=args, model=model, model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
                               training_data=train_dataset, collate_fn=train_dataset.collate_fn)
    # model_engine = model.cuda()
    #model_engine.load_checkpoint("NetQwen2-VL-7B-single3", 0)


    # Training setting
    max_epochs = 5
    train_loader_tqdm = tqdm(train_dataloader, ncols=120)

    best_test_loss = None
    print("Training start!")
    for epoch in range(max_epochs):
        model_engine.train()
        batch_loss = 0
        for i, data in enumerate(train_loader_tqdm):
            data = {k: v.to(model_engine.local_rank) for k, v in data.items()}
            output = model_engine(**data)
            loss = output['loss']
            batch_loss += loss.item()

            #optimizer.zero_grad()
            model_engine.backward(loss)
            #optimizer.clip_master_grads(1.0)
            model_engine.step()

            train_loader_tqdm.set_description(f'Training step {i}, train loss: {loss.item()}')

        print(f"Training epoch {epoch}: loss {batch_loss / len(train_dataloader)}")
        model_engine.save_checkpoint("NetQwen2-VL-7B-single3", epoch)

        model_engine.eval()
        labels, results = [], []
        for data in tqdm(test_dataloader):
            data = {k: v.to(model_engine.local_rank) for k, v in data.items()}
            with torch.no_grad():
                output = model_engine(**data)
            pred = output['pred'].detach().cpu()
            labels.append(data['labels'].cpu())
            results.append(pred)
            
        preds = torch.cat(results, dim=0).to('cpu')
        labels = torch.cat(labels, dim=0).to('cpu')
        acc_score, mse_score, mae_score = acc(preds, labels), mse(preds, labels), mae(preds, labels)
        print(f"Test epoch {epoch}: ACC: {acc_score}, MSE: {mse_score}, MAE: {mae_score}")
            