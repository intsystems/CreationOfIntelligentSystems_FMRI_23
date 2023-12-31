import os
from abc import ABC, abstractmethod
import torch
from transformers import BertConfig,BertPreTrainedModel, BertModel
from datetime import datetime
import torch.nn as nn

import torch
import torch.nn as nn
from collections import OrderedDict

def tuple_prod(x):
    prod = 1
    for xx in x:
        prod *= xx
    return prod

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,drop_rate=0.4):
        """
        green_block(inp, filters, name=None)
        ------------------------------------
        Implementation of the special residual block used in the paper. The block
        consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
        units, with a residual connection from the input `inp` to the output. Used
        internally in the model. Can be used independently as well.
        Note that images must come with dimensions "c, H, W, D"
        Parameters
        ----------
        `inp`: An keras.layers.layer instance, required
            The keras layer just preceding the green block.
        `out_channels`: integer, required
            No. of filters to use in the 3D convolutional block. The output
            layer of this green block will have this many no. of channels.
        Returns
        -------
        `out`: A keras.layers.Layer instance
            The output of the green block. Has no. of channels equal to `filters`.
            The size of the rest of the dimensions remains same as in `inp`.
        """
        super(GreenBlock, self).__init__()
        self.Drop_Rate = drop_rate
        # Define block
        self.block = nn.Sequential(OrderedDict([
            ('group_norm0', nn.GroupNorm(num_channels=in_channels, num_groups=in_channels // 4)),
            #('norm0', nn.BatchNorm3d(num_features=in_channels)),
            ('relu0', nn.LeakyReLU(inplace=True)),
            ('conv0', nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('group_norm1', nn.GroupNorm(num_channels=out_channels, num_groups=in_channels // 4)),
            #('norm1', nn.BatchNorm3d(num_features=out_channels)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('conv2', nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ]))

    def forward(self, inputs):
        #x_res = self.res(inputs)
        x_res = inputs
        #print('in green block: before')
        x = torch.nn.functional.dropout(self.block(inputs), p=self.Drop_Rate, training=self.training)
        #print('in green block: after')
        #return torch.cat([x, x_res], dim=1)
        return x + x_res



class UpGreenBlock(nn.Sequential):
    def __init__(self, in_features, out_features, shape, Drop_Rate):
        super(UpGreenBlock, self).__init__()

        self.add_module('conv', nn.Conv3d(in_features, out_features, kernel_size=1, stride=1))
        self.add_module('up', nn.Upsample(size=shape))
        self.add_module('green', GreenBlock(out_features, out_features, Drop_Rate))
class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.best_loss = 1000000
        self.best_accuracy = 0

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def determine_shapes(self,encoder,dim):
        def get_shape(module,input,output):
            module.input_shape = tuple(input[0].shape[-3:])
            module.output_shape = tuple(output[0].shape[-3:])
        hook1 = encoder.down_block1.register_forward_hook(get_shape)
        hook2 = encoder.down_block3.register_forward_hook(get_shape)
        input_shape = (1,2,) + dim  #batch,norms,H,W,D,time
        x = torch.ones((input_shape))
        with torch.no_grad():
            encoder(x)
            del x
        self.shapes = {'dim_0':encoder.down_block1.input_shape,
                       'dim_1':encoder.down_block1.output_shape,
                       'dim_2':encoder.down_block3.input_shape,
                       'dim_3':encoder.down_block3.output_shape}
        hook1.remove()
        hook2.remove()

    def register_vars(self,**kwargs):
        intermediate_vec = 2640
        if kwargs.get('task') == 'fine_tune':
            self.dropout_rates = {'input': 0, 'green': 0.35,'Up_green': 0,'transformer':0.1}
        else:
            self.dropout_rates = {'input': 0, 'green': 0.2, 'Up_green': 0.2,'transformer':0.1}

        self.BertConfig = BertConfig(hidden_size=intermediate_vec, vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=16, max_position_embeddings=30,
                                     hidden_dropout_prob=self.dropout_rates['transformer'])

        self.label_num = 1
        self.inChannels = 2
        self.outChannels = 1
        self.model_depth = 4
        self.intermediate_vec = intermediate_vec
        self.use_cuda = kwargs.get('cuda')
        self.shapes = kwargs.get('shapes')


    def load_partial_state_dict(self, state_dict,load_cls_embedding):
        print('loading parameters onto new model...')
        own_state = self.state_dict()
        loaded = {name:False for name in own_state.keys()}
        for name, param in state_dict.items():
            if name not in own_state:
                print('notice: {} is not part of new model and was not loaded.'.format(name))
                continue
            elif 'cls_embedding' in name and not load_cls_embedding:
                continue
            elif 'position' in name and param.shape != own_state[name].shape:
                print('debug line above')
                continue
            param = param.data
            own_state[name].copy_(param)
            loaded[name] = True
        for name,was_loaded in loaded.items():
            if not was_loaded:
                print('notice: named parameter - {} is randomly initialized'.format(name))


    def save_checkpoint(self, directory, title, epoch, optimizer=None,schedule=None):
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':self.state_dict(),
            'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
            'epoch':epoch}

        if schedule is not None:
            ckpt_dict['schedule_state_dict'] = schedule.state_dict()
            ckpt_dict['lr'] = schedule.get_last_lr()[0]
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path

        # Save the file with specific name
        core_name = title
        name = "{}_last_epoch.pth".format(core_name)
        torch.save(ckpt_dict, os.path.join(directory, name))


class Encoder(BaseModel):
    def __init__(self,**kwargs):
        super(Encoder, self).__init__()
        self.register_vars(**kwargs)
        self.down_block1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(self.inChannels, self.model_depth, kernel_size=3, stride=1, padding=1)),
            ('sp_drop0', nn.Dropout3d(self.dropout_rates['input'])),
            ('green0', GreenBlock(self.model_depth, self.model_depth, self.dropout_rates['green'])),
            ('downsize_0', nn.Conv3d(self.model_depth, self.model_depth * 2, kernel_size=3, stride=2, padding=1))]))
        self.down_block2 = nn.Sequential(OrderedDict([
            ('green10', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('green11', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('downsize_1', nn.Conv3d(self.model_depth * 2, self.model_depth * 4, kernel_size=3, stride=2, padding=1))]))
        self.down_block3 = nn.Sequential(OrderedDict([
            ('green20', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('green21', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('downsize_2', nn.Conv3d(self.model_depth * 4, self.model_depth * 8, kernel_size=3, stride=2, padding=1))]))
        self.final_block = nn.Sequential(OrderedDict([
            ('green30', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green31', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green32', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green33', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green']))]))

    def forward(self, x):
       # print('before')
        x = self.down_block1(x)
      #  print('after down_block1')
        x = self.down_block2(x)
      #  print('after down_block2')
        x = self.down_block3(x)
       # print('after down_block3')
        x = self.final_block(x)
      #  print('after final block')
        return x


class BottleNeck_in(BaseModel):
    def __init__(self,**kwargs):
        super(BottleNeck_in, self).__init__()
        self.register_vars(**kwargs)
        self.reduce_dimension = nn.Sequential(OrderedDict([
            ('group_normR', nn.GroupNorm(num_channels=self.model_depth * 8, num_groups=8)),
            # ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.LeakyReLU(inplace=True)),
            ('convR0', nn.Conv3d(self.model_depth * 8, self.model_depth // 2, kernel_size=(3, 3, 3), stride=1, padding=1)),
        ]))
        flat_factor = tuple_prod(self.shapes['dim_3'])
        self.flatten = nn.Flatten()
        if (flat_factor * self.model_depth // 2) == self.intermediate_vec:
            self.into_bert = nn.Identity()
            print('flattened vec identical to intermediate vector...\ndroppping fully conneceted bottleneck...')
        else:
            self.into_bert = nn.Linear(in_features=(self.model_depth // 2) * flat_factor, out_features=self.intermediate_vec)

    def forward(self, inputs):
        x = self.reduce_dimension(inputs)
        x = self.flatten(x)
        x = self.into_bert(x)

        return x


class BottleNeck_out(BaseModel):
    def __init__(self,**kwargs):
        super(BottleNeck_out, self).__init__()
        self.register_vars(**kwargs)
        flat_factor = tuple_prod(self.shapes['dim_3'])
        minicube_shape = (self.model_depth // 2,) + self.shapes['dim_3']
        self.out_of_bert = nn.Linear(in_features=self.intermediate_vec, out_features=(self.model_depth // 2) * flat_factor)
        self.expand_dimension = nn.Sequential(OrderedDict([
            ('unflatten', nn.Unflatten(1, minicube_shape)),
            ('group_normR', nn.GroupNorm(num_channels=self.model_depth // 2, num_groups=2)),
            # ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.LeakyReLU(inplace=True)),
            ('convR0', nn.Conv3d(self.model_depth // 2, self.model_depth * 8, kernel_size=(3, 3, 3), stride=1, padding=1)),
        ]))

    def forward(self, x):
        x = self.out_of_bert(x)
        return self.expand_dimension(x)

class Decoder(BaseModel):
    def __init__(self,**kwargs):
        super(Decoder, self).__init__()
        self.register_vars(**kwargs)
        self.decode_block = nn.Sequential(OrderedDict([
            ('upgreen0', UpGreenBlock(self.model_depth * 8, self.model_depth * 4, self.shapes['dim_2'], self.dropout_rates['Up_green'])),
            ('upgreen1', UpGreenBlock(self.model_depth * 4, self.model_depth * 2, self.shapes['dim_1'], self.dropout_rates['Up_green'])),
            ('upgreen2', UpGreenBlock(self.model_depth * 2, self.model_depth, self.shapes['dim_0'], self.dropout_rates['Up_green'])),
            ('blue_block', nn.Conv3d(self.model_depth, self.model_depth, kernel_size=3, stride=1, padding=1)),
            ('output_block', nn.Conv3d(in_channels=self.model_depth, out_channels=self.outChannels, kernel_size=1, stride=1))
        ]))

    def forward(self, x):
        x = self.decode_block(x)
        return x


class AutoEncoder(BaseModel):
    def __init__(self,dim,**kwargs):
        super(AutoEncoder, self).__init__()
        # ENCODING
        self.task = 'autoencoder_reconstruction'
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder,dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)

        # BottleNeck out of bert
        self.from_bert = BottleNeck_out(**kwargs)

        # DECODER
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        if x.isnan().any():
            print('nans in data!')
        batch_size, Channels_in, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, Channels_in, W, H, D)
        encoded  = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = self.from_bert(encoded)
        reconstructed_image = self.decoder(encoded)
        _, Channels_out, W, H, D = reconstructed_image.shape
        reconstructed_image = reconstructed_image.reshape(batch_size, T, Channels_out, W, H, D).permute(0, 2, 3, 4, 5, 1)
        return {'reconstructed_fmri_sequence': reconstructed_image}


class Transformer_Block(BertPreTrainedModel, BaseModel):
    def __init__(self,config,**kwargs):
        super(Transformer_Block, self).__init__(config)
        self.register_vars(**kwargs)
        self.cls_pooling = True
        self.bert = BertModel(self.BertConfig, add_pooling_layer=self.cls_pooling)
        self.init_weights()
        self.cls_embedding = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.BertConfig.hidden_size), nn.LeakyReLU())
        self.register_buffer('cls_id', torch.ones((kwargs.get('batch_size'), 1, self.BertConfig.hidden_size)) * 0.5,persistent=False)


    def concatenate_cls(self, x):
        cls_token = self.cls_embedding(self.cls_id)
        return torch.cat([cls_token, x], dim=1)


    def forward(self, x ):
        inputs_embeds = self.concatenate_cls(x=x)
        outputs = self.bert(input_ids=None,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=self.BertConfig.use_return_dict
                            )

        sequence_output = outputs[0][:, 1:, :]
        pooled_cls = outputs[1]

        return {'sequence': sequence_output, 'cls': pooled_cls}


class Encoder_Transformer_Decoder(BaseModel):
    def __init__(self, dim,**kwargs):
        super(Encoder_Transformer_Decoder, self).__init__()
        self.task = 'transformer_reconstruction'
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder,dim)
        kwargs['shapes'] = self.shapes

        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig, **kwargs)

        # BottleNeck out of bert
        self.from_bert = BottleNeck_out(**kwargs)

        # DECODER
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        out = transformer_dict['sequence'].reshape(batch_size * T, -1)
        out = self.from_bert(out)
        reconstructed_image = self.decoder(out)
        reconstructed_image = reconstructed_image.reshape(batch_size, T, self.outChannels, W, H, D).permute(0, 2, 3, 4, 5, 1)
        return {'reconstructed_fmri_sequence': reconstructed_image}

