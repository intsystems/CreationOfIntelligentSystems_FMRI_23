from trainer import Trainer
if __name__ == "__main__":
    args = {'lr_init':1e-3, 'lr_gamma':0.97, 'lr_step':1000, 'seq_len':5,
            'weight_decay':1e-7, 'task':'transformer_reconstruction', 'cuda':True,
            'transformer_hidden_layers':2, 'batch_size':8,
            'reconstruction_factor':5, 'perceptual_factor':1, 'intensity_factor':1,
            'title':'Transformer', 'directory':'TFF_weights',
            'nEpochs':1, 'memory_constraint':0.1, 'loaded_model_weights_path': None}
    trainer = Trainer(**args)
    history_losses = trainer.training()