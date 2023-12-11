from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from model import Encoder_Transformer_Decoder
from utils import *
from datasets import fMRIDataset, fMRIVideoDataset
class LrHandler():
    def __init__(self,**kwargs):
        self.final_lr = 1e-5
        self.base_lr = kwargs.get('lr_init')
        self.gamma = kwargs.get('lr_gamma')
        self.step_size = kwargs.get('lr_step')


    def set_lr(self,dict_lr):
        if self.base_lr is None:
            self.base_lr = dict_lr

    def set_schedule(self,optimizer):
        self.schedule = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

    def schedule_check_and_update(self):
        if self.schedule.get_last_lr()[0] > self.final_lr:
            self.schedule.step()

def get_intense_voxels(yy,shape):
    y = yy.clone()
    low_quantile, high_quantile, = (0.9,0.99)
    voxels = torch.empty(shape)
    for batch in range(y.shape[0]):
        for TR in range(y.shape[-1]):
            yy = y[batch, :, :, :, TR]
            background = yy[0, 0, 0]
            yy[yy <= background] = 0
            yy = abs(yy)
            voxels[batch, :, :, :, :, TR] = (yy > torch.quantile(yy[yy > 0], low_quantile)).unsqueeze(0)
    return voxels.view(shape)>0

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        #h_relu_3_3 = h
        out = (h_relu_1_2, h_relu_2_2)
        return out

class Percept_Loss(nn.Module):
    def __init__(self, memory_constraint: float, use_cuda=False):
        super(Percept_Loss, self).__init__()
        print('notice: changed layers in perceptual back to old version')
        self.memory_constraint = memory_constraint
        self.vgg = Vgg16()
        if use_cuda:
            self.vgg.cuda()

        self.loss = nn.MSELoss()

    def forward(self, input, target):
        assert input.shape == target.shape, 'input and target should have identical dimension'
        assert len(input.shape) == 6
        batch, channel, width, height, depth, T = input.shape
        num_slices = batch * T * depth
        represent = torch.randperm(num_slices)[:int(num_slices * self.memory_constraint)]
        input = input.permute(0, 5, 1, 4, 2, 3).reshape(num_slices, 1, width, height)
        target = target.permute(0, 5, 1, 4, 2, 3).reshape(num_slices, 1, width, height)
        input = input[represent, :, :, :].repeat(1,3,1,1)
        target = target[represent, :, :, :].repeat(1,3,1,1)

        input = self.vgg(input)
        target = self.vgg(target)
        loss = 0
        for i,j in zip(input,target):
            loss += self.loss(i,j)
        return loss


class Trainer():
    """
    main class to handle training, validation and testing.
    note: the order of commands in the constructor is necessary
    """
    def __init__(self,**kwargs):
        self.register_args(**kwargs)
        self.lr_handler = LrHandler(**kwargs)
        if 'video' in self.task.lower():
          dataset = fMRIVideoDataset(train=True, seq_len=self.seq_len,
                                     video_path=self.video_path, skip_frames=self.skip_frames)
        else:
          dataset = fMRIDataset(seq_len=self.seq_len)
        self.train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.create_model()
        self.initialize_weights(load_cls_embedding=False)
        self.create_optimizer()
        self.lr_handler.set_schedule(self.optimizer)

        if 'vae' in self.task.lower():
            self.use_KL = True
        else:
            self.use_KL = False
            self.KL_factor = 0

        self.losses = {'intensity':
                           {'is_active':True,'factor':self.intensity_factor},
                       'perceptual':
                           {'is_active':True, 'factor':self.perceptual_factor},
                       'reconstruction':
                           {'is_active':True,'factor':self.reconstruction_factor},
                       'KL':
                           {'is_active':self.use_KL,'factor':self.KL_factor}}
        self.reconstruction_loss_func = nn.L1Loss()
        self.perceptual_loss_func = Percept_Loss(self.memory_constraint, self.cuda)
        self.intensity_loss_func = nn.L1Loss() #(thresholds=[0.9, 0.99]

    def initialize_weights(self,load_cls_embedding):
        if self.loaded_model_weights_path is not None:
            state_dict = torch.load(self.loaded_model_weights_path)
            self.lr_handler.set_lr(state_dict['lr'])
            self.model.load_partial_state_dict(state_dict['model_state_dict'], load_cls_embedding)

    def save_model(self):
        epoch = self.nEpochs
        self.model.save_checkpoint(self.directory, self.title, epoch,
                                      optimizer=self.optimizer,schedule=self.lr_handler.schedule)


    def create_optimizer(self):
        lr = self.lr_handler.base_lr
        params = self.model.parameters()
        weight_decay = self.kwargs.get('weight_decay')
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def create_model(self):
        dim = (40, 64, 64)
        if self.task.lower() == 'transformer_reconstruction':
            self.model = Encoder_Transformer_Decoder(dim,**self.kwargs)
        if self.cuda:
            self.model = self.model.cuda()

    def training(self):
        loss_history = {name:[] for name, current_loss_dict in self.losses.items() if current_loss_dict['is_active']}
        for epoch in range(self.nEpochs):
            epoch_losses = self.train_epoch(epoch)
            for name in loss_history.keys():
              loss_history[name] += epoch_losses[name]
              print(name + f' loss {epoch_losses[name][-1]}')

            print('______epoch summary {}/{}_____\n'.format(epoch+1,self.nEpochs))
            #print(f'intensity loss {int_loss[-1]}, perceptual loss {perc_loss[-1]}, reconstruction loss {rec_loss[-1]}')
        return loss_history


    def train_epoch(self,epoch):
        self.train()
        epoch_losses = {name:[] for name, current_loss_dict in self.losses.items() if current_loss_dict['is_active']}

        for batch_idx, input_dict in enumerate(tqdm(self.train_loader)):
           # self.writer.total_train_steps += 1
            self.optimizer.zero_grad()
            loss_dict, loss = self.forward_pass(input_dict)

            for name in epoch_losses.keys():
                epoch_losses[name].append(loss_dict[name])

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            loss.backward()
            self.optimizer.step()
            self.lr_handler.schedule_check_and_update()
           # if batch_idx % 20 == 0:
           #     print(loss_dict)

        return epoch_losses

    def train(self):
        self.mode = 'train'
        self.model = self.model.train()

    def forward_pass(self,input_dict):
        #input_dict = {k:(v.cuda() if self.cuda else v) for k,v in input_dict.items()}
        if self.cuda:
            if isinstance(input_dict, dict):
                input_dict = {k:(v.cuda() if self.cuda else v) for k,v in input_dict.items()}
            else:
                input_dict = input_dict.cuda()

        output_dict = self.model(input_dict)
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)
        #if self.task == 'fine_tune':
        #    self.compute_accuracy(input_dict, output_dict)
        return loss_dict, loss


    def aggregate_losses(self,input_dict,output_dict):
        final_loss_dict = {}
        final_loss_value = 0
        if isinstance(input_dict, dict):
          input = input_dict['fmri_seq']
        else:
          input = input_dict
        for loss_name, current_loss_dict in self.losses.items():
            if current_loss_dict['is_active']:
                loss_func = getattr(self, 'compute_' + loss_name)
                current_loss_value = loss_func(input, output_dict)
                if current_loss_value.isnan().sum() > 0:
                    warnings.warn('found nans in computation')
                    print('at {} loss'.format(loss_name))
                lamda = current_loss_dict['factor']
                factored_loss = current_loss_value * lamda
                final_loss_dict[loss_name] = factored_loss.item()
                final_loss_value += factored_loss

        final_loss_dict['total'] = final_loss_value.item()
        return final_loss_dict, final_loss_value

    def compute_KL(self,input_dict,output_dict):
        """
            This function returns the value of KL(p1 || p2),
            where p1 = Normal(mean_1, exp(log_std_1)**2), p2 is standard normal distribution.
            Note that we consider the case of diagonal covariance matrix.
        """
        mean_1 = output_dict['mean_1']
        log_std_1 = output_dict['log_std_1']

        if 'mean_2' in output_dict.keys():
          mean_2 = output_dict['mean_2']
          log_std_2 = output_dict['log_std_2']
        else:
          mean_2 = torch.zeros_like(mean_1)
          log_std_2 = torch.zeros_like(log_std_1)

        var_1 = torch.exp(log_std_1) ** 2
        var_2 = torch.exp(log_std_2) ** 2

        KL = 2*(log_std_2 - log_std_1) - mean_1.shape[-1] + (mean_1 - mean_2)**2 / var_2 + var_1 / var_2
        return torch.mean(KL / 2)

    def compute_reconstruction(self,input_dict,output_dict):
        fmri_sequence = input_dict[:,0].unsqueeze(1)
        reconstruction_loss = self.reconstruction_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return reconstruction_loss

    def compute_intensity(self,input_dict,output_dict):
        per_voxel = input_dict[:,1,:,:,:,:]
        voxels = get_intense_voxels(per_voxel, output_dict['reconstructed_fmri_sequence'].shape)
        output_intense = output_dict['reconstructed_fmri_sequence'][voxels]
        truth_intense = input_dict[:,0][voxels.squeeze(1)]
        intensity_loss = self.intensity_loss_func(output_intense.squeeze(), truth_intense)
        return intensity_loss

    def compute_perceptual(self,input_dict,output_dict):
        fmri_sequence = input_dict[:,0].unsqueeze(1)
        perceptual_loss = self.perceptual_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return perceptual_loss

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs