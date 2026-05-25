import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.init as init


class pBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=torch.device('cpu'),
                 weight_scale=0.1, rho_offset=0.1, zeta=10):
        super(pBNN, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()  
        self.num_layers = 2
        self.layer_param_shapes = self.get_layer_param_shapes()
        self.mus = nn.ParameterList() 
        self.rhos = nn.ParameterList()
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sigmas = torch.tensor([1.] * len(self.layer_param_shapes), device=self.device)

        for i, shape in enumerate(self.layer_param_shapes):
            if i % 2 == 0:  
                self.batch_norms.append(nn.BatchNorm1d(shape[0]))  
            if i % 2 == 0:  
                self.dropout_layers.append(nn.Dropout(0.2))  
            
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale * torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)
            
    def get_layer_param_shapes(self):
        layer_param_shapes = []
        for i in range(self.num_layers + 1):
            if i == 0:
                W_shape = (self.input_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            elif i == self.num_layers:
                W_shape = (self.hidden_dim, self.output_dim)
                b_shape = (self.output_dim,)
            else:
                W_shape = (self.hidden_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            layer_param_shapes.extend([W_shape, b_shape])
        return layer_param_shapes

    def transform_rhos(self, rhos):
        return [F.softplus(rho) for rho in rhos]

    # Reparameterization Trick
    def transform_gaussian_samples(self, mus, rhos, epsilons):
        # compute softplus for variance
        self.sigmas = self.transform_rhos(rhos)
        samples = []
        for j in range(len(mus)): samples.append(mus[j] + self.sigmas[j] * epsilons[j])
        return samples
    
    # collect a group noise from standard normal distribution
    def sample_epsilons(self, param_shapes):
        # mean is zero, std is 0.001
        epsilons = [torch.normal(mean=torch.zeros(shape), std=0.01*torch.ones(shape)).to(self.device) for shape in
                    param_shapes]
        return epsilons
#-------------------------------uncertainty prediction----------------------------
    def net(self, X, layer_params):
        layer_input = X
        for i in range(len(layer_params) // 2 - 1):
            layer_input = torch.mm(layer_input, layer_params[2 * i]) + layer_params[2 * i + 1]
            layer_input = F.relu(layer_input)
            
        
        output = torch.mm(layer_input, layer_params[-2]) + layer_params[-1]
        
        return output
    
    # compute the log_likelihood to evaluate the matching degree between true and predict value
    def log_softmax_likelihood(self, mean_output, y):
        return torch.nansum(y * F.log_softmax(mean_output, dim=0), dim=0)

    
    def hellinger_distance_normal(self, mu1, sigma1, mu2, sigma2):
        term1 = torch.sqrt(2 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
        term2 = torch.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))
        return torch.sqrt(1 - term1 * term2)
    
    # loss function of client
    def local_loss(self, personal_mean_output, personal_log_var_output, label_one_hot, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        log_likelihood_sum = torch.sum(self.log_softmax_likelihood(personal_mean_output, label_one_hot))
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i], sigmas[i]),
                            Normal(mus_local[i].detach(), sigmas_local[i].detach())))  for i in range(len(params))])
        
        last_layer_idx = -1
        hellinger_dist_sum = torch.sum(self.hellinger_distance_normal(
            mus[last_layer_idx], 
            sigmas[last_layer_idx],
            mus_local[last_layer_idx].detach(), 
            sigmas_local[last_layer_idx].detach()
        ))
        
        lamda = 1/(hellinger_dist_sum.item()+0.5)
        lamda = int(lamda) if lamda.is_integer() else math.ceil(lamda)
        print('models_lamda:',lamda)

        return  1.0 / num_batches * (10 * KL_q_w) - log_likelihood_sum
    
    # loss function of server
    def global_loss(self, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i].detach(), sigmas[i].detach()),
                        Normal(mus_local[i], sigmas_local[i]))) for i in range(len(params))])
        
        last_layer_idx = -1
        hellinger_dist_sum = torch.sum(self.hellinger_distance_normal(
            mus[last_layer_idx].detach(),
            sigmas[last_layer_idx].detach(),
            mus_local[last_layer_idx],
            sigmas_local[last_layer_idx]
        ))
        
        lamda = 1/(hellinger_dist_sum.item()+0.1)
        lamda = int(lamda) if lamda.is_integer() else math.ceil(lamda)

        return 1.0 / num_batches * (KL_q_w)  
    
class pSBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=torch.device('cpu'),
                 weight_scale=0.1, rho_offset=0.1, zeta=10, sparse_lambda=10):
        super(pSBNN, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()  # Dropout layers
        self.num_layers = 1
        self.layer_param_shapes = self.get_layer_param_shapes()
        self.mus = nn.ParameterList() 
        self.rhos = nn.ParameterList()
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sparse_lambda = sparse_lambda  # Regularization for sparsity
        self.sigmas = torch.tensor([1.] * len(self.layer_param_shapes), device=self.device)

        for i, shape in enumerate(self.layer_param_shapes):
            if i % 2 == 0:  # Only add BatchNorm for weight layers (even indices)
                self.batch_norms.append(nn.BatchNorm1d(shape[0]))  # Adjust the dimensions based on your layer's shape
                
            if i % 2 == 0:  # Apply dropout only after weight layers
                self.dropout_layers.append(nn.Dropout(0.2))  # Dropout with 50% probability
        
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale * torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)
            
    def get_layer_param_shapes(self):
        layer_param_shapes = []
        for i in range(self.num_layers + 1):
            if i == 0:
                W_shape = (self.input_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            elif i == self.num_layers:
                W_shape = (self.hidden_dim, self.output_dim)
                b_shape = (self.output_dim,)
            else:
                W_shape = (self.hidden_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            layer_param_shapes.extend([W_shape, b_shape])
        return layer_param_shapes

    def transform_rhos(self, rhos):
        return [F.softplus(rho) for rho in rhos]

    def transform_gaussian_samples(self, mus, rhos, epsilons):
        self.sigmas = self.transform_rhos(rhos)
        samples = []
        for j in range(len(mus)): samples.append(mus[j] + self.sigmas[j] * epsilons[j])
        return samples
    
    def sample_epsilons(self, param_shapes):
        epsilons = [torch.normal(mean=torch.zeros(shape), std=0.01*torch.ones(shape)).to(self.device) for shape in param_shapes]
        return epsilons

    def net(self, X, layer_params):
        layer_input = X
        for i in range(len(layer_params) // 2 - 1):
            layer_input = torch.mm(layer_input, layer_params[2 * i]) + layer_params[2 * i + 1]
            layer_input = F.leaky_relu(layer_input)
        
        output = torch.mm(layer_input, layer_params[-2]) + layer_params[-1]
        return output
    
    def log_softmax_likelihood(self, mean_output, y):
        return torch.nansum(y * F.log_softmax(mean_output, dim=0), dim=0)
        
    def hellinger_distance_normal(self, mu1, sigma1, mu2, sigma2):
        term1 = torch.sqrt(2 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
        term2 = torch.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))
        return torch.sqrt(1 - term1 * term2)

    def local_loss(self, personal_mean_output, personal_log_var_output, label_one_hot, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        log_likelihood_sum = torch.sum(self.log_softmax_likelihood(personal_mean_output, label_one_hot))
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i], sigmas[i]),
                            Normal(mus_local[i].detach(), sigmas_local[i].detach()))) for i in range(len(params))])
        
        # Regularization for sparsity (L1 penalty on weights)
        sparse_penalty = 0
        for mu in self.mus:
            sparse_penalty += torch.sum(torch.abs(mu))  # L1 penalty on weights
        
        hellinger_dist_sum = sum(torch.stack([torch.sum(self.hellinger_distance_normal(mus[i], sigmas[i],
                            mus_local[i].detach(), sigmas_local[i].detach()))
                             for i in range(len(params))]))
        
        return 1.0 / num_batches * (self.zeta * KL_q_w) - log_likelihood_sum + self.sparse_lambda * sparse_penalty
    
    def global_loss(self, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i].detach(), sigmas[i].detach()),
                        Normal(mus_local[i], sigmas_local[i]))) for i in range(len(params))])
        
        hellinger_dist_sum = sum(torch.stack([torch.sum(self.hellinger_distance_normal(mus[i].detach(), sigmas[i].detach(),
                            mus_local[i], sigmas_local[i]))
                             for i in range(len(params))]))
        
        return 1.0 / num_batches * (self.zeta * KL_q_w) + self.sparse_lambda * torch.sum(torch.cat([mu.view(-1) for mu in mus])) 
 
    
class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()

        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc4 = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        return x
    


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        residual = x  
        out = self.fc1(x)
        out = self.fc2(out)
        out += residual 
        
        return out

class ResNet(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=200, output_dim=1, num_blocks=3):
        super(ResNet, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x


