from trainer import Trainer
import matplotlib.pyplot as plt

def sample_fMRI(trainer, num = 2):
    i = 0
    n = num
    for batch in trainer.train_loader:
        if i == n:
            break
        out = trainer.model(batch.cuda())['reconstructed_fmri_sequence']
        out = out.cpu().detach().numpy()
        plt.subplot(2*n,2,2*i + 2)
        plt.imshow(out[0,0,0,:,:,0])

        plt.axis('off')
        if i == 0:
        
            plt.title('Predicted slice')
    

        plt.subplot(2*n,2, 2*i + 1)
        plt.imshow(batch[0, 0,0,:,:,0])
        if i == 0:
            plt.title('GT next slice')
        plt.axis('off')
        i += 1
    plt.savefig('slices.pdf')
    

def draw_plots(history_losses):

    plt.semilogy(history_losses['intensity'], label='intensity')
    plt.semilogy(history_losses['reconstruction'], label='reconstruction')
    plt.semilogy(history_losses['perceptual'], label='perceptual')
    plt.legend()
    plt.ylabel('metric')
    plt.xlabel('iterations')
    plt.savefig('training_loss.pdf')

if __name__ == "__main__":
    args = {'lr_init':1e-3, 'lr_gamma':0.97, 'lr_step':1000, 'seq_len':5,
            'weight_decay':1e-7, 'task':'transformer_reconstruction', 'cuda':True,
            'transformer_hidden_layers':2, 'batch_size':8,
            'reconstruction_factor':5, 'perceptual_factor':1, 'intensity_factor':1,
            'title':'Transformer', 'directory':'TFF_weights',
            'nEpochs':1, 'memory_constraint':0.1, 'loaded_model_weights_path': None}
    trainer = Trainer(**args)
    history_losses = trainer.training()

    draw_plots(history_losses)
    sample_fMRI(trainer)

    
    