import torch


loss_mse = torch.nn.MSELoss()
loss_mae = torch.nn.L1Loss()

def acc(pds, gts):
    pds = torch.round(pds) 
    acc = torch.mean((pds == gts).float())

    return acc.item()



def mse(pds, gts):
    mes_loss = loss_mse(pds, gts)
    return mes_loss.item()



def mae(pds, gts):
    mae_loss = loss_mae(pds, gts)
    return mae_loss.item()
  

def compute_metrics(output):
    results = {}
    pred_logits = None

    return results


if __name__ == '__main__':
    pds = torch.tensor([1.2, 2.3, 3.6, 4.8, 5.5, 6.0])
    gts = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int)
    acc(pds, gts)
    mse(pds, gts)
    mae(pds, gts)
