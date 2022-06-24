import torch.nn as nn
import torch
import torch.nn.functional as F
from ohem_ce_loss import OhemCELoss
import torch.cuda.amp as amp
from lr_scheduler import WarmupPolyLrScheduler
from meters import TimeMeter, AvgMeter
import utils
import numpy as np



def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


def loss_function (S, T) :

  T = torch.mean(T, axis=1)
  S = torch.mean(S, axis=1)

  return (T-S).pow(2).mean()


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': 5e-3 * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': 5e-3 * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=5e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )
    return optim


def set_meters():
    time_meter = TimeMeter(150000)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(4)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters







class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """
  ############################### teacher added #####################
    def __init__(self, model, teacher, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.distill_loss = sim_dis_compute
        self.metric = metric
        self.device = device

        self.loss = 0.0
        self.distillation_loss = 0.0
        ############## teacher ##################
        self.teacher = teacher


    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        self.teacher.eval()


        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            

            [outputs, feature_maps] = self.model(inputs)

            with torch.no_grad():
              [teacher_out, T_f] = self.teacher(inputs)

            T_f.detach()

            self.optim.zero_grad()

            ##########################################
           # T = 1
           # distill_loss = 10 * nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_out/T, dim=1))
           #### loss = self.criterion(outputs, labels) + 10 * distill_loss 
            #########################################
            

            TF = T_f
            SF = feature_maps

            loss = 0.0
            temp = self.criterion(outputs, labels)
            self.loss = temp.item()
            loss = loss + temp

            temp2 = loss_function(TF, SF)
            self.distillation_loss = temp2.item()
            loss = loss + temp2

           # loss = loss + distill_loss
           # print(outputs.shape)
            
            
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()



            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                    print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
          

        return epoch_loss / len(self.data_loader), self.metric.value() 
