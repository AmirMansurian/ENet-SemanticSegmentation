import torch.nn as nn
import torch
import torch.nn.functional as F
from ohem_ce_loss import OhemCELoss
import torch.cuda.amp as amp
from lr_scheduler import WarmupPolyLrScheduler
from meters import TimeMeter, AvgMeter



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
        self.metric = metric
        self.device = device
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
        criteria_pre = OhemCELoss(0.7)
        criteria_aux = [OhemCELoss(0.7) for _ in range(4)]

        optim = set_optimizer(self.model)

        ## mixed precision training
        scaler = amp.GradScaler()


        ## meters
        time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

        ## lr scheduler
        lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
            max_iter=150000, warmup_iter=1000,
            warmup_ratio=0.1, warmup='exp', last_epoch=-1,)



        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
           # outputs = self.model(inputs)[0]
            
           

            ########################### teacher output ###########################################
            #with torch.no_grad():
            #  teacher_out = self.teacher(inputs)
            
            #distill_loss = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_out, dim=1))
            ######################################################################################

           # print(outputs)
            #print(teacher_out)

















            optim.zero_grad()
            with amp.autocast(enabled=True):
                logits, *logits_aux = self.model(inputs)
                loss_pre = criteria_pre(logits, labels)
                loss_aux = [crit(lgt, labels) for crit, lgt in zip(criteria_aux, logits_aux)]
                loss = loss_pre + sum(loss_aux)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            torch.cuda.synchronize()

            time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]










            # Loss computation
            #loss = self.criterion(outputs, labels)
            #print('#####################' + str(loss)+'###################')
            #################### changed #########################
            #loss = 10*distill_loss
            #print('#####################' + str(loss2)+'###################')
            ######################################################


            # Backpropagation
           # self.optim.zero_grad()
            #loss.backward()
            #self.optim.step()

            # Keep track of loss for current epoch
            #epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(logits.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return loss / len(self.data_loader), self.metric.value()
