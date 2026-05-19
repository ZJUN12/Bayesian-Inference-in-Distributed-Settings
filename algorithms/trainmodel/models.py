import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.init as init
import torch

# personalized bayesian neural network
class pBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=torch.device('cpu'),
                 weight_scale=0.1, rho_offset=0.1, zeta=10):
        super(pBNN, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()  # Dropout layers
        self.num_layers = 2
        self.layer_param_shapes = self.get_layer_param_shapes()
        # nn.ParameterList() is a part of torch.nn.module, which can be used to store parameters
        # store mean value parameters
        self.mus = nn.ParameterList() 
        # store variance parameters
        self.rhos = nn.ParameterList()
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sigmas = torch.tensor([1.] * len(self.layer_param_shapes), device=self.device)


        for i, shape in enumerate(self.layer_param_shapes):
            if i % 2 == 0:  # Only add BatchNorm for weight layers (even indices)
                self.batch_norms.append(nn.BatchNorm1d(shape[0]))  # Adjust the dimensions based on your layer's shape
                
            # if i % 2 == 0:  # Apply dropout only after weight layers
            #     self.dropout_layers.append(nn.Dropout(0.2))  # Dropout with 50% probability
            
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale * torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)

        target_initial_bias = 0 
        target_initial_weight = 0.0

        with torch.no_grad():
            self.rhos[-2].data.fill_(-7.0) # scenario II
            # self.rhos[-2].data.fill_(-5.0) # scenario I
            self.mus[-2].data.fill_(target_initial_weight)
            self.mus[-1].data.fill_(target_initial_bias)
            

    # a function that can be used to get the layer parameters shapes
    def get_layer_param_shapes(self):
        # initialize a empty list: this can be used to store parameters
        # parameters of each layer include weight matrix (W) and bias vector (b)
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
        # F.softplus is a activation function, it can map to a range of non-negtive
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
        # epsilons = [torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)).to(self.device) for shape in
        #             param_shapes]
        epsilons = [torch.randn(shape, device=self.device)
            for shape in param_shapes]
        return epsilons
#-------------------------------uncertainty prediction----------------------------
    def net(self, X, layer_params):
        if X.ndim > 2:
            layer_input = X.view(X.size(0), -1) 
        else:
            layer_input = X

        for i in range(len(layer_params) // 2 - 1):
            layer_input = torch.mm(layer_input, layer_params[2 * i]) + layer_params[2 * i + 1]
            layer_input = F.tanh(layer_input)
            
        output = torch.mm(layer_input, layer_params[-2]) + layer_params[-1]
        mean = output[:, 0].unsqueeze(1)
        log_var = output[:, 1].unsqueeze(1)
        
        return mean, log_var
    
    # compute the log_likelihood to evaluate the matching degree between true and predict value
    def log_softmax_likelihood(self, mean_output, y):
        return torch.nansum(y * F.log_softmax(mean_output, dim=0), dim=0)
        
    
    def hellinger_distance_normal(self, mu1, sigma1, mu2, sigma2):
        term1 = torch.sqrt(2 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
        term2 = torch.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))
        return torch.sqrt(1 - term1 * term2)


    def local_loss(self, personal_mean_output, personal_log_var_output, label, params, mus, sigmas, mus_local, sigmas_local, num_batches):
            if label.ndim == 1:
                label = label.view(-1, 1)
                
            personal_log_var_output = torch.clamp(personal_log_var_output, min=-10, max=10)
            sigma_sq = torch.exp(personal_log_var_output)
            log_likelihood = -0.5 * torch.sum(torch.log(sigma_sq) + (label - personal_mean_output)**2 / sigma_sq)
            KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i], sigmas[i]),
                                Normal(mus_local[i].detach(), sigmas_local[i].detach())))  for i in range(len(params))])
            
            # Hellinger Distance (保持不变)
            last_layer_idx = -1
            hellinger_dist_sum = torch.sum(self.hellinger_distance_normal(
                mus[last_layer_idx], 
                sigmas[last_layer_idx],
                mus_local[last_layer_idx].detach(), 
                sigmas_local[last_layer_idx].detach()
            ))
            
            lamda = 1 / (hellinger_dist_sum + 0.5)
            
            # 返回总 Loss (-log_likelihood 变成了 NLL)
            return 1.0 / num_batches * (lamda * KL_q_w) - log_likelihood   
            # return 1.0 / num_batches * (KL_q_w) - log_likelihood 

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
        
        # print('models_hellinger_dist_sum:',1/(hellinger_dist_sum.item()+0.1))
        lamda = 1/(hellinger_dist_sum.item()+0.1)
        lamda = int(lamda) if lamda.is_integer() else math.ceil(lamda)
        # return 1.0 / num_batches * (self.zeta * KL_q_w)  ### Contribution: personalized weight Zeta
        return 1.0 / num_batches * (KL_q_w)  ### Contribution: personalized weight Zeta
    
    # def train(self, epochs):
    #     LOSS = 0
    #     self.model.train(epochs)
    #     for epoch in range(1, self.local_epochs + 1):
    #         self.model.train()
    #         temp_model = copy.deepcopy(list(self.model.parameters()))

    #         # === Step 1 ===
    #         X, y = self.get_next_train_batch()
    #         self.optimizer.zero_grad()
    #         output = self.model(X)
    #         loss = self.loss(output, y)
    #         loss.backward()
            
    #         # [建议] Step 1 也要防止内循环走太远，可以加裁剪
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10) 
    #         self.optimizer.step()

    #         # === Step 2 ===
    #         X, y = self.get_next_train_batch()
    #         self.optimizer.zero_grad()
    #         output = self.model(X)
    #         loss = self.loss(output, y)
    #         loss.backward()

    #         # restore the model parameters
    #         for old_p, new_p in zip(self.model.parameters(), temp_model):
    #             old_p.data = new_p.data.clone()
            
    #         # ==========================================
    #         # 【关键修改】 在这里加入梯度裁剪！救命的一行！
    #         # 防止 Step 2 算出来的梯度过大，把原始模型带崩
    #         # ==========================================
    #         total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            
    #         # 可选：打印一下看看梯度是不是很大
    #         # if total_norm > 20: print(f"Gradient clipped! Norm: {total_norm}")

    #         self.optimizer.step(beta = self.beta)
            
    #         # clone model
    #         self.clone_model_paramenter(self.model.parameters(), self.local_model)

    #     return LOSS


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import copy  # 修复可能出现的 copy 未定义错误

class pSBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=torch.device('cpu'),
                 weight_scale=0.1, rho_offset=-3.0, zeta=0.0, sparse_lambda=0.0):
        """
        初始化参数说明:
        rho_offset: 设为 -3.0 左右，确保初始标准差 sigma 较小 (约0.05)，防止初始噪声过大导致梯度失效。
        zeta, sparse_lambda: 建议初始设为 0.0，先让模型学会拟合数据，后期再慢慢增加。
        """
        super(pSBNN, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.num_layers = 1
        self.layer_param_shapes = self.get_layer_param_shapes()
        
        # 存储贝叶斯参数
        self.mus = nn.ParameterList() 
        self.rhos = nn.ParameterList()
        
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        
        # 将正则系数转为 Tensor 并放入正确设备
        self.zeta = torch.tensor(zeta, device=self.device).float()
        self.sparse_lambda = torch.tensor(sparse_lambda, device=self.device).float()
        
        self.dropout_layers = nn.ModuleList()

        # --- 参数初始化 ---
        for i, shape in enumerate(self.layer_param_shapes):
            # 仅在权重层（偶数索引）之后添加 Dropout
            if i % 2 == 0: 
                self.dropout_layers.append(nn.Dropout(0.1)) # Dropout 设小一点，如 0.1

            # 初始化 Mu: 使用 Xavier 初始化或者较小的随机数，防止梯度消失
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale * torch.ones(shape)))
            
            # 初始化 Rho: 决定初始方差的大小
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            
            self.mus.append(mu)
            self.rhos.append(rho)
        
        # 【关键修复】在 __init__ 结束前初始化 self.sigmas，防止报错
        with torch.no_grad():
            self.sigmas = self.transform_rhos(self.rhos)

    def get_layer_param_shapes(self):
        layer_param_shapes = []
        for i in range(self.num_layers + 1):
            if i == 0:
                W_shape = (self.input_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            elif i == self.num_layers:
                # 输出层: 输出维度 * 2 (前一半是 Mean，后一半是 LogVar)
                W_shape = (self.hidden_dim, self.output_dim * 2) 
                b_shape = (self.output_dim * 2,)
            else:
                W_shape = (self.hidden_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            layer_param_shapes.extend([W_shape, b_shape])
        return layer_param_shapes

    def transform_rhos(self, rhos):
        # Softplus: log(1 + exp(rho))，保证 sigma 始终 > 0
        return [F.softplus(rho) for rho in rhos]

    def transform_gaussian_samples(self, mus, rhos, epsilons):
        # Reparameterization Trick: w = mu + sigma * epsilon
        self.sigmas = self.transform_rhos(rhos) # 实时更新 sigma
        samples = []
        for j in range(len(mus)): 
            samples.append(mus[j] + self.sigmas[j] * epsilons[j])
        return samples
    
    def sample_epsilons(self, param_shapes):
        # 生成标准高斯噪声
        epsilons = [torch.normal(mean=torch.zeros(shape), std=1.0*torch.ones(shape)).to(self.device) for shape in param_shapes]
        return epsilons

    def net(self, X, layer_params):
        layer_input = X
        # layer_params 排列: [W0, b0, W1, b1, ..., W_out, b_out]
        num_hidden_layers = len(layer_params) // 2 - 1
        
        for i in range(num_hidden_layers):
            W = layer_params[2 * i]
            b = layer_params[2 * i + 1]
            
            layer_input = torch.mm(layer_input, W) + b
            
            if i < len(self.dropout_layers):
                 layer_input = self.dropout_layers[i](layer_input)
            
            layer_input = F.leaky_relu(layer_input)
        
        # 输出层
        output = torch.mm(layer_input, layer_params[-2]) + layer_params[-1]
        
        # 拆分 Mean 和 Log Variance
        mean = output[:, :self.output_dim]
        log_var = output[:, self.output_dim:]
        
        # 【数值稳定】限制 log_var 范围，防止方差过大导致 Loss 变成 NaN
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        
        return mean, log_var

    def forward(self, X):
        """标准前向传播 (用于推理/测试)"""
        epsilons = self.sample_epsilons(self.layer_param_shapes)
        layer_params = self.transform_gaussian_samples(self.mus, self.rhos, epsilons)
        return self.net(X, layer_params)
    
    # === 损失函数 ===
    
    def gaussian_nll(self, mean, log_var, y):
        """
        高斯负对数似然 (回归任务专用)
        Loss = 0.5 * (log(var) + (y - mean)^2 / var)
        """
        if y.shape != mean.shape:
            y = y.view_as(mean)
            
        variance = torch.exp(log_var)
        # 加 1e-6 防止除零
        loss = 0.5 * (log_var + (y - mean)**2 / (variance + 1e-6))
        return torch.sum(loss) # Sum over batch

    def local_loss(self, personal_mean, personal_log_var, y, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        # 1. 设备检查与同步
        device = personal_mean.device 
        if y.device != device:
            y = y.to(device)

        # 2. 似然损失 (NLL)
        nll_loss = self.gaussian_nll(personal_mean, personal_log_var, y)
        
        # 3. KL 散度 (如果 zeta=0 则跳过计算，节省时间)
        KL_q_w = torch.tensor(0.0, device=device)
        if self.zeta > 0:
            kl_list = []
            for i in range(len(params)):
                curr_device = mus[i].device
                
                # 强制把参考分布的参数搬运到 curr_device，并切断梯度
                mu_target = mus_local[i].to(curr_device).detach()
                sigma_target = sigmas_local[i].to(curr_device).detach()
                
                p = Normal(mus[i], sigmas[i])
                q = Normal(mu_target, sigma_target)
                
                kl_list.append(torch.sum(kl_divergence(p, q)))
            KL_q_w = sum(kl_list)
        
        # 4. 稀疏正则 (L1) - 必须加 Abs
        sparse_penalty = torch.tensor(0.0, device=device)
        if self.sparse_lambda > 0:
            for mu in self.mus:
                sparse_penalty += torch.sum(torch.abs(mu)) 
        
        # 确保系数在正确设备
        if self.zeta.device != device: self.zeta = self.zeta.to(device)
        if self.sparse_lambda.device != device: self.sparse_lambda = self.sparse_lambda.to(device)
            
        # 组合 Loss
        total_loss = nll_loss + (1.0 / num_batches) * self.zeta * KL_q_w + self.sparse_lambda * sparse_penalty
        
        return total_loss
    
    def global_loss(self, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        """全局损失 (用于 pFedMe 更新)"""
        # 设备处理同上，略微简化
        kl_list = []
        for i in range(len(params)):
            curr_device = mus[i].device
            mu_target = mus_local[i].to(curr_device)
            sigma_target = sigmas_local[i].to(curr_device)
            
            p = Normal(mus[i].detach(), sigmas[i].detach())
            q = Normal(mu_target, sigma_target)
            
            kl_list.append(torch.sum(kl_divergence(p, q)))
            
        KL_q_w = sum(kl_list) if kl_list else torch.tensor(0.0, device=mus[0].device)
        
        l1_reg = torch.sum(torch.cat([torch.abs(mu).view(-1) for mu in mus]))
        
        # 确保系数在正确设备
        zeta_val = self.zeta.to(mus[0].device)
        
        return 1.0 / num_batches * (zeta_val * KL_q_w) + self.sparse_lambda * l1_reg
    
class DNN(nn.Module):
    def __init__(self, input_dim = 1, mid_dim = 150, output_dim = 1):
        super(DNN, self).__init__()

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
        # self.bn = nn.BatchNorm1d(input_dim)
        # self.fc2 = nn.Linear(mid_dim, mid_dim)
        # self.fc3 = nn.Linear(mid_dim, mid_dim)
        self.fc4 = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        # output = F.relu(x)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义残差块（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        residual = x  # 保存输入作为残差连接
        out = self.fc1(x)
        out = self.fc2(out)
        out += residual  # 添加残差连接
        
        return out

class ResNet(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=200, output_dim=1, num_blocks=3):
        super(ResNet, self).__init__()
        # 输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # 残差块
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 输入层
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        # 残差块
        x = self.residual_blocks(x)
        # 输出层
        x = self.output_layer(x)
        # print("models_output:",x)
        return x


