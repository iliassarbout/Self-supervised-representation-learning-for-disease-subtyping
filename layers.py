import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from sklearn.mixture import GaussianMixture
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'





    
class Encoder(nn.Module):
    def __init__(self, z_dim, n = 4, shape = [3139, 3217, 3105, 383],weight = [0.25, 0.25, 0.25, 0.25],sample=False):
        super().__init__()
        self.latent_dim = z_dim
        self.n = n
        self.sample = sample
        self.shape = shape
        self.weight = weight
        self.encoding_dim = int(self.latent_dim)
        self.dims = [max(int(self.encoding_dim * weight[i]),1) for i in range(n)] #to avoid 0
        self.X = nn.ModuleList([nn.Linear(shape[i], 20*self.dims[i]) for i in range(n)])
        self.X2 = nn.ModuleList([nn.Linear(20*self.dims[i], self.dims[i]) for i in range(n)])
        if n > 1:
            self.concat = nn.Linear(sum(self.dims), self.encoding_dim)
        else:
            self.concat = nn.Linear(self.dims[0], self.encoding_dim)
            
        self.bn = nn.BatchNorm1d(self.latent_dim,momentum=0.01, eps=0.001)#same hyper parameters as keras
        self.act = nn.GELU()
        
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim) #new
        self.bn2 = nn.BatchNorm1d(self.latent_dim,momentum=0.01, eps=0.001) #new

        
        self.z_mean = nn.Linear(self.encoding_dim, self.encoding_dim)
        self.z_log_var = nn.Linear(self.encoding_dim, self.encoding_dim)
        
    def sampling(self, z_mean, z_log_var,eps = None):
        std = torch.exp(0.5 * z_log_var)
        if eps is None:
            eps = torch.randn_like(std)
        return z_mean + std * eps
        
    def forward(self, x):
        x = [self.X[i](x[i]) for i in range(self.n)]
        x = [self.X2[i](x[i]) for i in range(self.n)]
        if self.n > 1:
            x = torch.cat(x, dim=1)
        else:
            x = x[0]
        x = self.concat(x)
        x = self.bn(x)
        x = self.act(x)
        
        x = self.act(self.bn2(self.fc2(x)))
        
        z_mean = self.z_mean(x)
        z_log_var = F.softplus(self.z_log_var(x))
        if self.sample:
            z = self.sampling(z_mean, z_log_var)
            return z,z_mean,z_log_var
        else:
            return z_mean,z_log_var

    
class Decoder(nn.Module):
    def __init__(self, z_dim, n = 4, shape = [3139, 3217, 3105, 383],weight = [0.25, 0.25, 0.25, 0.25]):
        super().__init__()
        self.latent_dim = z_dim
        self.n = n
        self.shape = shape
        self.weight = weight
        self.denses = nn.ModuleList()
        self.denses.append(nn.Linear(self.latent_dim, self.latent_dim))
        self.denses.append(nn.BatchNorm1d(self.latent_dim,momentum=0.01, eps=0.001))#same hyper parameters as keras
        self.denses.append(nn.GELU())
        #self.denses.append(nn.GELU())
        
        self.dec_modules = nn.ModuleList()
        
        for i in range(self.n):
            self.dec_modules.append(nn.Linear(self.latent_dim, self.shape[i]))
        
        
    def forward(self, x):
        out = x
        for layer in self.denses:
            out = layer(out)
        new_shape = torch.Size(torch.cat((torch.tensor([self.n]),torch.tensor(out.shape)))) #replicate n times
        out = out.expand(new_shape)
        out = [self.dec_modules[i](out[i]) for i in range(self.n)]
        return out

class DecoderDisc(nn.Module):
    def __init__(self, latent_dim, n, shape, weight):
        super().__init__()
        self.latent_dim = latent_dim
        self.n = n
        self.shape = shape
        self.weight = weight
        self.denses = nn.ModuleList()
        self.denses.append(nn.Linear(self.latent_dim, self.latent_dim))
        self.denses.append(nn.BatchNorm1d(self.latent_dim,momentum=0.01, eps=0.001))#same hyper parameters as keras
        self.denses.append(nn.GELU())
        #self.denses.append(nn.GELU())
        
        self.decode = nn.ModuleList()
        
        for i in range(self.n):
            self.decode.append(nn.Linear(self.latent_dim, self.shape[i]))
        
        self.disc = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), #new
            nn.GELU(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid()
        )
        self.decode.append(self.disc)
        
    def forward(self, x):
        out = x
        for layer in self.denses:
            out = layer(out)
        new_shape = torch.Size(torch.cat((torch.tensor([self.n+1]),torch.tensor(out.shape)))) #replicate n times and once more for discriminator
        out = out.expand(new_shape)
        out = [self.decode[i](out[i]) for i in range(self.n+1)]
        return out
    