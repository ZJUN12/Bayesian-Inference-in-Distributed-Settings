import copy
from torch.autograd import Variable
from algorithms.trainmodel.models import *
from algorithms.users.userbase import User
import numpy as np

class UserpFedBayes(User):
    def __init__(self, K, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, personal_learning_rate, device, output_dim=10):
        super().__init__(K, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, device, output_dim=output_dim)
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.personal_learning_rate = personal_learning_rate
        self.optimizer1 = torch.optim.Adam(self.personal_model.parameters(), lr=self.personal_learning_rate)  
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_loaders = []
        for unit_samples in train_data:
            if len(unit_samples) > 0:
                loader = torch.utils.data.DataLoader(
                    unit_samples, batch_size=batch_size, shuffle=True)
                self.train_loaders.append(loader)
            else:
                self.train_loaders.append(None)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        self.personal_model.train()
        K_MC_Samples = 5  
        for epoch in range(1, epochs + 1):
            for unit_idx, loader in enumerate(self.train_loaders):
                if loader is None: continue
                for batch_idx, (X, Y) in enumerate(loader):
                    X, Y = X.to(self.device), Y.to(self.device)
                    if len(Y.shape) > 2: Y = Y.squeeze(-1)

                    self.optimizer1.zero_grad()

                    batch_loss_personal = 0
                    
                    for _ in range(K_MC_Samples):
                        epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
                        layer_params1 = self.personal_model.transform_gaussian_samples(
                            self.personal_model.mus, self.personal_model.rhos, epsilons)
                        
                        prediction_mean, prediction_var = self.personal_model.net(X, layer_params1)
                        loss_step = self.personal_model.local_loss(
                             prediction_mean,  
                             prediction_var,
                             Y, layer_params1,
                             self.personal_model.mus, self.personal_model.sigmas,
                             copy.deepcopy(self.model.mus),
                             [t.clone().detach() for t in self.model.sigmas], 
                             self.local_epochs
                        )
                        batch_loss_personal += loss_step

                    batch_loss_personal /= K_MC_Samples
                    batch_loss_personal.backward()
                    self.optimizer1.step()
                    self.optimizer2.zero_grad()
                    batch_loss_global = 0
                    
                    for _ in range(K_MC_Samples):
                        epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
                        layer_params2 = self.model.transform_gaussian_samples(self.model.mus, self.model.rhos, epsilons)
                        global_loss = self.model.global_loss(
                            [t.clone().detach() for t in layer_params1],
                            copy.deepcopy(self.personal_model.mus),
                            [t.clone().detach() for t in self.personal_model.sigmas],
                            self.model.mus, self.model.sigmas, self.local_epochs)
                        
                        batch_loss_global += global_loss

                    batch_loss_global /= K_MC_Samples
                    batch_loss_global.backward()
                    self.optimizer2.step()

        return LOSS
