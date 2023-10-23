import argparse
import torch
from torchvision import datasets, transforms
import torchvision
import numpy as np
from sklearn.decomposition import PCA
ngf = 64
ndf = 64
lookahead = 5
prior_alpha = torch.tensor(0.01)
prior_beta = torch.tensor(0.02)
glogit_prior_mu = torch.Tensor([-1.6])
prior_sigma = torch.Tensor([1.])


init_weight_mean_var = (0, .001)
uniform_low = torch.Tensor([.01])
uniform_high = torch.Tensor([.99])

parser = argparse.ArgumentParser(description='Dir-VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', 
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)') 
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--category', type=int, default=10, metavar='K',
                    help='the number of categories in the dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



def beta_func(a, b):
    return (torch.lgamma(a).exp() + torch.lgamma(b).exp()) / torch.lgamma(a + b).exp()


def logistic_func(x):
    return 1 / (1 + torch.exp(-x))
parametrizations = dict(Kumar='Kumaraswamy', GLogit='Gauss_Logit', gaussian='Gaussian', GEM='GEM', Dir='Dirichlet_dist', GD='GDVAE', gdwo='GDWO')

def dataDir(analysis =False, MNIST = False, FashionMNIST = False, KMNIST=False, USPS=False, CIFAR10_ = False, CIFAR10 = False, Country211=False, SVHN = False):
    if MNIST:
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.ToTensor())
        test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
        
        input_shape = list(train_dataset.data[0].shape)
        input_ndims = np.prod(input_shape)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
        nc = 1
    elif FashionMNIST:
        train_dataset = datasets.FashionMNIST('../data_FashionMNIST', train=True, download=True,
                           transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST('../data_FashionMNIST', train=False, transform=transforms.ToTensor())
        
        input_shape = list(train_dataset.data[0].shape)
        input_ndims = np.prod(input_shape)
        train_loader = torch.utils.data.DataLoader(
        
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
        nc = 1
        
    elif KMNIST:
        train_dataset = datasets.KMNIST('../data_KMNIST', train=True, download=True,
                           transform=transforms.ToTensor())
        test_dataset = datasets.KMNIST('../data_KMNIST', download=True,
                                       transform=transforms.ToTensor())
        
        input_shape = list(train_dataset.data[0].shape)
        input_ndims = np.prod(input_shape)
        train_loader = torch.utils.data.DataLoader(
        
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
        nc = 1
        
    elif USPS:
        train_dataset = datasets.USPS('../data_USPS', train=True, download=True,
                           transform=torchvision.transforms.Compose([transforms.Resize(size=(28, 28)),
                                     torchvision.transforms.ToTensor()
                                     ]))
        test_dataset = datasets.USPS('../data_USPS', download=True,
                                     transform=torchvision.transforms.Compose([transforms.Resize(size=(28, 28)),
                                     torchvision.transforms.ToTensor()
                                     ]))

        input_shape = list(train_dataset.data[0].shape)
        input_ndims = np.prod(input_shape)
        train_loader = torch.utils.data.DataLoader(
        
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
        nc = 1
        
    elif CIFAR10_:
                # Load the CIFAR10 dataset
        trainset = torchvision.datasets.CIFAR10(root='../data_CIFAR10', train=True,
                                                download=True, transform=transforms.ToTensor())
        
        # Compute the mean and standard deviation of the dataset
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=False)
        data = next(iter(trainloader))[0]
        mean = data.mean(axis=(0, 2, 3))
        std = data.std(axis=(0, 2, 3))
        
        # Define the normalization transform
        normalize = transforms.Normalize(mean=mean, std=std)
        
        train_dataset = datasets.CIFAR10('../data_CIFAR10', train=True, download=True,
                           transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(), normalize,
                           torchvision.transforms.Resize(28)
                           ]))
        test_dataset = datasets.CIFAR10('../data_CIFAR10', train=False,transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(), normalize,
                           torchvision.transforms.Resize(28)
                           ]))
        
        input_shape = list(train_dataset.data[0].shape)
        input_ndims = np.prod(input_shape)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
        nc = 3
    
        
    elif SVHN:

        train_dataset = datasets.SVHN('../data_SVHN', split = 'train', download=True,
                           transform = transforms.Compose([
            transforms.Pad(padding=2),
            transforms.RandomCrop(size=(28, 28)),
            transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
            transforms.ToTensor(),
            #transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ]))
            
            
        test_dataset = datasets.SVHN('../data_SVHN', split= 'test',download=True,
                           transform = transforms.Compose([
            transforms.Pad(padding=2),
            transforms.RandomCrop(size=(28, 28)),
            transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
            transforms.ToTensor(),
            #transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ]))
        
        input_shape = list(train_dataset.data[0].shape)
        input_ndims = np.prod(input_shape)
      
        #train_dataset = torch.stack(tuple(
        #torch.pca_lowrank(train_dataset[i][0])
        #for i in range(train_dataset.data.shape[0])
        #), dim=0)
        
        #U, S, V = torch.pca_lowrank(torch.as_tensor(train_dataset) , q=None, center=True, niter=3)
        #train_dataset = torch.matmul(train_dataset, V[:, :28])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
        nc = 3
        
    elif CIFAR10:
        
        train_dataset = datasets.CIFAR10('../data_CIFAR10', train=True, download=True,
                           transform=torchvision.transforms.Compose([transforms.RandomCrop(size=(28, 28)), 
                           torchvision.transforms.RandomHorizontalFlip(),
                           torchvision.transforms.ToTensor()
                           #transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
                           ]))
        test_dataset = datasets.CIFAR10('../data_CIFAR10', train=False,
                           transform=torchvision.transforms.Compose([transforms.RandomCrop(size=(28, 28)),
                           torchvision.transforms.ToTensor()
                           #transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
                           ]))
        
        input_shape = list(train_dataset.data[0].shape)
        input_ndims = np.prod(input_shape)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
        nc = 3
        
    elif Country211:
        
        train_dataset = datasets.Country211('../data_Country211', split='train', download=True,
                           transform=torchvision.transforms.Compose([transforms.RandomCrop(size=(28, 28)), 
                           torchvision.transforms.RandomHorizontalFlip(),
                           torchvision.transforms.ToTensor()
                           #transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
                           ]))
        test_dataset = datasets.Country211('../data_Country211', split='test', download=True,
                           transform=torchvision.transforms.Compose([transforms.RandomCrop(size=(28, 28)),
                           torchvision.transforms.ToTensor()
                           #transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
                           ]))
        
        input_shape = list(train_dataset.data[0].shape)
        input_ndims = np.prod(input_shape)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
        nc = 3
    
    else:
        print("Data directory not specified")
        
    if analysis:
        return train_dataset, test_dataset, input_shape, input_ndims, nc
    else :
        return train_loader, test_loader, input_shape, input_ndims, nc

train_loader, test_loader, input_shape, input_ndims, nc = dataDir(FashionMNIST = True)