import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from utils import parser, parametrizations, lookahead, train_loader, test_loader, nc
from VAEs import StickBreakingVAE
import time

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model_path = 'trained_models'
time_now = datetime.datetime.now().__format__('%b_%d_%Y_%H_%M')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)  


parametrization = parametrizations['GD'] #Best error is 95
#parametrization = parametrizations['gdwo'] 
#parametrization = parametrizations['Dir'] #Best error is 133.43, epoch 82
#parametrization = parametrizations['GEM'] # epoch 98 Best reconstruction error is 205.83880883789064
#parametrization = parametrizations['Kumar']
#parametrization = parametrizations['GLogit']
#parametrization = parametrizations['gaussian']

model = StickBreakingVAE(parametrization).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3) #3
#scheduler_redPlat = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
#scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
#scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20,30, 40,60,80], gamma=0.05)

parametrization_str = parametrization if model._get_name() == "StickBreakingVAE" else ''
model_name = '_'.join(filter(None, [model._get_name(), parametrization_str]))
start_epoch = 1



def prior(K, alpha):
    """
    Prior for the model.
    :K: number of categories
    :alpha: Hyper param of Dir
    :return: mean and variance tensors
    """
    # ラプラス近似で正規分布に近似
    # Approximate to normal distribution using Laplace approximation
    a = torch.Tensor(1, K).float().fill_(alpha)
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t() # Parameters of prior distribution after approximation

# init save directories
#tb_writer = SummaryWriter(f'logs/{model_name}')
if not os.path.exists(os.path.join(model_path, model_name)):
    os.mkdir(os.path.join(model_path, model_name))
best_test_epoch = None
best_test_loss = None
best_test_model = None
best_test_optimizer = None
stop_training = None
best_reconstruction_loss = None


rec_loss_train = []
KL_loss_train = []
rec_loss_test = []
KL_loss_test = []



def train(epoch, data, train_loss, train_recon_loss, train_KL_loss):
    data = data.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    
    reconLoss, analytical_kld = model.ELBO_loss(recon_batch, data, mu, logvar, model.kl_divergence,  model.parametrization)
    loss = reconLoss + analytical_kld
    loss = loss.mean()
    loss.backward()
    train_loss += loss.item()
    train_recon_loss += reconLoss.mean().item()
    train_KL_loss += analytical_kld.mean().item()
    optimizer.step()
    #scheduler_1.step()
    #scheduler_2.step()
    if batch_idx % args.log_interval == 0:
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.item() / len(data)))
        
    return recon_batch, train_loss, train_recon_loss, train_KL_loss


    

def test(epoch, data, test_loss, test_recon_loss, test_KL_loss):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        recon_batch_test, mu, logvar = model(data)
        reconLoss, analytical_kld = model.ELBO_loss(recon_batch_test, data, mu, logvar, model.kl_divergence,  model.parametrization)
        loss = reconLoss + analytical_kld
        test_loss += loss.mean().item()
       
        test_recon_loss += reconLoss.mean().item()
        test_KL_loss += analytical_kld.mean().item()
        if i == 0:
            n = min(data.size(0), 18)
            comparison = torch.cat([data[:n],
                                  recon_batch_test.view(args.batch_size, nc, 28, 28)[:n]])
            save_image(comparison.cpu(),
                     'image/recon_' + str(model_name) + '.png', nrow=n)
    return recon_batch_test, test_loss, test_recon_loss, test_KL_loss

inference_time = 0
for epoch in range(1, args.epochs + 1):
    train_loss = 0
    train_recon_loss = 0
    train_KL_loss = 0
    for batch_idx, (data, Y) in enumerate(train_loader):        
        
        if len(data) %  args.batch_size == 0:
            recon_batch, train_loss, train_recon_loss, train_KL_loss = train(epoch, data, train_loss, train_recon_loss, train_KL_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
      epoch, train_loss / len(train_loader.dataset)))
    train_recon_loss /= len(train_loader.dataset)
    train_KL_loss /= len(train_loader.dataset)
    rec_loss_train.append(train_recon_loss)
    KL_loss_train.append(train_KL_loss)
    #tb_writer.add_scalar(f"{time_now}/Loss/train", train_loss, epoch)
    #tb_writer.add_scalar(f"{time_now}/Regularization_Loss/train", train_KL_loss, epoch)
    #tb_writer.add_scalar(f"{time_now}/Reconstruction_Loss/train", train_recon_loss, epoch)
    
    ############################# TESTING #################################
    test_loss = 0
    test_recon_loss = 0
    test_KL_loss = 0
    
    for i, (data, label) in enumerate(test_loader):
        if len(data) %  args.batch_size == 0:
            recon_batch_test, test_loss, test_recon_loss, test_KL_loss = test(epoch, data, test_loss, test_recon_loss, test_KL_loss)        
    
    test_loss /= len(test_loader.dataset)
    test_recon_loss /= len(test_loader.dataset)
    test_KL_loss /= len(test_loader.dataset)
    rec_loss_test.append(test_recon_loss)
    KL_loss_test.append(test_KL_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    if  epoch == start_epoch:
        best_test_epoch = epoch
        best_test_loss = test_loss
        best_reconstruction_loss = test_recon_loss
    else:
        best_test_epoch = epoch if test_loss < best_test_loss else best_test_epoch
        best_test_loss = test_loss if best_test_epoch else best_test_loss
        stop_training = True if epoch - best_test_epoch > lookahead else False
        best_reconstruction_loss = test_recon_loss if best_test_epoch else best_reconstruction_loss
    
    
    ################### UPDATE MODEL PARAMETERS If THERE'S IMPROVEMENT ##################
    if epoch == best_test_epoch:
        with torch.no_grad():
            sample = torch.randn(64, args.category).to(device)
            sample = torch.div(sample,torch.reshape(torch.sum(sample,1), (-1, 1)))
            start_time = time.time()
            sample = model.decode(sample).cpu()
            inference_time = time.time() - start_time
            print('The inference time is : {}'.format(inference_time))
            img_tensor=sample.view(64, nc, 28, 28)
            #tb_writer.add_images(f'{64}_random_latent_space_samples_{time_now}',
                        # img_tensor,
                         #global_step=epoch,
                         #dataformats='NCHW')
            
            save_image(img_tensor,'image/sample_' + str(model_name) + '.png')
            
        # save trained weights
        print("===> updating model with best parameters")
        best_test_model = model.state_dict().copy()
        best_test_optimizer = optimizer.state_dict().copy()
    elif stop_training:
        print("===> Training stopped because there was no further improvement")
        break
        
print('The inference time is : {}'.format(inference_time))    
print('Best epoch is ' + str (best_test_epoch))
print('Best reconstruction error is ' + str(best_reconstruction_loss))
#tb_writer.close()
np.save('outResults/rec_loss_train_' + str(model_name), np.array(rec_loss_train))
np.save('outResults/KL_loss_train_' + str(model_name), np.array(KL_loss_train))
np.save('outResults/rec_loss_test_' + str(model_name), np.array(rec_loss_test))
np.save('outResults/KL_loss_test_' + str(model_name), np.array(KL_loss_test))
#print(rec_loss_test)
#print('#' * 30)
#print(KL_loss_test)
torch.save({'epoch': best_test_epoch,
            'model_state_dict': best_test_model,
            'optimizer_state_dict': best_test_optimizer},
           os.path.join(model_path, model_name, f'best_checkpoint_{model_name}_{time_now}'))