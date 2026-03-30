import time
import warnings
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import json
import pickle
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from tqdm import tqdm
import dgl
import numpy as np

from RGCN import RGCN
from Classifier import Classifier
from EarlyStopping import EarlyStopping
from utils import set_random_seed, convert_to_gpu, get_optimizer_and_lr_scheduler, get_node_data_loader, get_n_params
from metrics import *


args = {'dataset': 'cross_plat_short_videos',
        'model_name': 'RGCN',
        'embedding_name': 'bert_THLM',
        'mode': 'train',
        'seed': 2,
        'predict_category': 'video',
        'cuda': 0,
        'learning_rate': 1e-5,
        'hidden_units': [1024, 1024],
        'n_layers': 2,
        'dropout': 0.0,
        'n_bases': -1,
        'use_self_loop': True,
        'batch_size': 512,
        'node_neighbors_min_num': 50,
        'optimizer': 'adam', 'weight_decay': 1e-4, 'epochs': 100, 'patience': 10}


args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'
device = torch.cuda.set_device('cuda:{}'.format(args["cuda"]))
# device = 'cpu'


def load_dgl_data(args):
    with open('data/graph_with_labels.pkl', 'rb') as f:
        data = pickle.load(f)

    g = data['graph']
    labels = torch.tensor(list(data['labels'].values()))

    train_idx = torch.nonzero(g.nodes['video'].data['train_mask']).squeeze()
    test_idx = torch.nonzero(g.nodes['video'].data['test_mask']).squeeze()

    return g, labels, args['predict_category'], train_idx, test_idx, args['batch_size'], args['epochs'], args['patience']

def evaluate(model: nn.Module, loader: dgl.dataloading.DataLoader, 
             labels: torch.Tensor, predict_category: str, device: str, mode: str):
    """

    :param model: model
    :param loader: data loader (test)
    :param loss_func: loss function
    :param labels: node labels
    :param predict_category: str
    :param device: device str
    :param mode: str, evaluation mode, test
    :return:
    """
    model.eval()
    with torch.no_grad():
        y_trues = []
        y_predicts = []
        #total_loss = 0.0
        loader_tqdm = tqdm(loader, ncols=120)
        for batch, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            # target node relation representation in the heterogeneous graph
            input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}

            nodes_representation = model[0](blocks, copy.deepcopy(input_features))

            y_predict = model[1](nodes_representation[predict_category])

            # Tensor, (samples_num, )
            y_true = convert_to_gpu(labels[output_nodes[predict_category]], device=device)

            #loss = loss_func(y_predict, y_true)

            #total_loss += loss.item()
            y_trues.append(y_true.detach().cpu())
            y_predicts.append(y_predict.detach().cpu())
            

            loader_tqdm.set_description(f'{mode} for the {batch}-th batch')

       #total_loss /= (batch + 1)
        y_trues = torch.cat(y_trues, dim=0)
        y_predicts = torch.cat(y_predicts, dim=0)

        ACC = acc(y_predicts, y_trues)
        MSE = mse(y_predicts, y_trues)
        MAE = mae(y_predicts, y_trues)

    return ACC, MSE, MAE, y_trues, y_predicts


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    #torch.set_num_threads(1)

    set_random_seed(args['seed'])

    print(f'loading dataset {args["dataset"]}...')

    with torch.no_grad():
        graph, labels, target_node_type, train_idx, test_idx, batch_size, num_epochs, patience = load_dgl_data(args)
    print(f'get node data loader...')

    train_loader, test_loader = get_node_data_loader(args['node_neighbors_min_num'], args['n_layers'],
                                                     graph,
                                                     batch_size=args['batch_size'],
                                                     sampled_node_type=args['predict_category'],
                                                     train_idx=train_idx, 
                                                     test_idx=test_idx)

    rgcn = RGCN(graph=graph, input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                hidden_sizes=args['hidden_units'], num_bases=args['n_bases'], dropout=args['dropout'],
                use_self_loop=args['use_self_loop'])

    classifier = Classifier(n_hid=args['hidden_units'][-1])

    model = nn.Sequential(rgcn, classifier)

    model = convert_to_gpu(model, device=args['device'])
    print(model)

    print(f'Model #Params: {get_n_params(model)}.')

    print(f'configuration is {args}')

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model, args['optimizer'], args['learning_rate'],
                                                          args['weight_decay'],
                                                          steps_per_epoch=len(train_loader), epochs=args['epochs'])

    save_model_folder = f"./save_model/{args['dataset']}/{args['model_name']}/{args['embedding_name']}"

    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                   save_model_name=args['model_name'])

    loss_func = torch.nn.SmoothL1Loss()

    train_steps = 0
    s = time.time()
    if args['mode'] == 'train':
        for epoch in range(args['epochs']):
            model.train()
            
            train_y_trues = []
            train_y_predicts = []
            train_total_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, ncols=120)
            for batch, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):
                blocks = [convert_to_gpu(b, device=args['device']) for b in blocks]
                # target node relation representation in the heterogeneous graph
                input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}

                nodes_representation= model[0](blocks, copy.deepcopy(input_features))

                train_y_predict = model[1](nodes_representation[args['predict_category']])

                # Tensor, (samples_num, )
                train_y_true = convert_to_gpu(labels[output_nodes[args['predict_category']]], device=args['device'])
                loss = loss_func(train_y_predict.float(), train_y_true.float())

                train_total_loss += loss.item()
                train_y_trues.append(train_y_true.detach().cpu())
                train_y_predicts.append(train_y_predict.detach().cpu())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

                train_loader_tqdm.set_description(f'training for the {batch}-th batch, train loss: {loss.item()}')
                # step should be called after a batch has been used for training.
                train_steps += 1
                scheduler.step(train_steps)

            train_total_loss /= (batch + 1)
            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)
            train_acc = acc(train_y_predicts, train_y_trues)
            train_mse = mse(train_y_predicts, train_y_trues)
            train_mae = mae(train_y_predicts, train_y_trues)

            print(f'Epoch: {epoch + 1}, train acc: {train_acc:.4f}, train mse: {train_mse:.4f}, train mae: {train_mae:.4f}, train loss: {train_total_loss:.4f}')                                                             

            model.eval()
            test_acc, test_mse, test_mae, test_y_trues, test_y_predicts = evaluate(model, loader=test_loader,
                                                                                   labels=labels,
                                                                                   predict_category=args['predict_category'],
                                                                                   device=args['device'],
                                                                                   mode='test')                                 
            print(f'Epoch: {epoch + 1}, test acc: {test_acc:.4f}, test mse: {test_mse:.4f}, test mae: {test_mae:.4f}')

            early_stop = early_stopping.step([('ALL', test_acc - test_mse - test_mae, True)], model)

            if early_stop:
                break
        
    param_path = f"./save_model/{args['dataset']}/{args['model_name']}/{args['embedding_name']}/{args['model_name']}.pkl"
    params = torch.load(param_path, map_location='cpu')
    model.load_state_dict(params)
    model = convert_to_gpu(model, device=args['device'])
    early_stopping.load_checkpoint(model)

    model.eval()

    nodes_representation = model[0].inference(graph, copy.deepcopy(
        {ntype: graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}), device=args['device'])
    embed_path = f"./save_emb/{args['dataset']}/"
    if not os.path.exists(embed_path):
        os.makedirs(embed_path, exist_ok=True)
    torch.save(nodes_representation,
               f"./save_emb/{args['dataset']}/{args['embedding_name']}_{args['model_name']}.pkl")

    train_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        train_idx]
    train_y_trues = convert_to_gpu(labels[train_idx], device=args['device'])
    train_acc = acc(train_y_predicts, train_y_trues)  
    train_mse = mse(train_y_predicts, train_y_trues)
    train_mae = mae(train_y_predicts, train_y_trues)                                                                                                                       

    test_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        test_idx]
    test_y_trues = convert_to_gpu(labels[test_idx], device=args['device'])  
    test_acc = acc(test_y_predicts, test_y_trues)
    test_mse = mse(test_y_predicts, test_y_trues)
    test_mae = mae(test_y_predicts, test_y_trues)                                                              


    # save model result
    result_json = {
            "train accuracy": train_acc,
            "train mean_absolute_error": train_mse,
            "train mean_squared_error": train_mae,
            "test accuracy": test_acc,
            "test mean_absolute_error": test_mse,
            "test mean_squared_error": test_mae
        }
    result_json = json.dumps(result_json, indent=4)

    save_result_folder = f"./results/{args['dataset']}/{args['model_name']}"
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args['embedding_name']}.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json)

    print(result_json)
    res = {"labels": test_y_trues.tolist(),
           "predicts": test_y_predicts.tolist(),
           "idx": test_idx.tolist()}
    with open(os.path.join(save_result_folder, f"{args['model_name']}_test.json"), 'w') as file:
        file.write(json.dumps(res, indent=4))
    sys.exit()
