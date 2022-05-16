import torch.nn as nn
import torch
import torch.nn.functional as F

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
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            outputs = self.model(inputs)
            
           

            ########################### teacher output ###########################################
            with torch.no_grad():
              teacher_out = self.teacher(inputs)
            
            #distill_loss = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_out, dim=1))
            ######################################################################################

            print(outputs)
            print(teacher_out)





            # Loss computation
            loss = self.criterion(outputs, teacher_out)
            #print('#####################' + str(loss)+'###################')
            #################### changed #########################
            #loss = 10*distill_loss
            #print('#####################' + str(loss2)+'###################')
            ######################################################


            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
