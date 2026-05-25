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
        self.N_Batch = len(train_data) // batch_size
        self.personal_learning_rate = personal_learning_rate
        self.optimizer1 = torch.optim.Adam(self.personal_model.parameters(), lr=self.personal_learning_rate, weight_decay=1e-4)
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # self.optimizer1 = torch.optim.SGD(self.personal_model.parameters(), lr=self.personal_learning_rate, momentum=0.9, weight_decay=1e-2)
        # self.optimizer2 = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-2)
    def set_grads(self, new_grads):
        # judge if new_grads are the type of nn.Parameter
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        # judge if new_grads are the type of nn.Parameter
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        N_Samples = 20
        self.model.train()
        self.personal_model.train()
        personal_outputs = []
        local_outputs = []
        #-------------------Contribution: Distributed online collaborative RUL prediction by SGVI-----------------------------
        for epoch in range(1, epochs + 1):
            for s in range(1, N_Samples + 1):
                X, Y = self.get_next_train_batch()
                # print('userpFedbayes_X_shape:',X.shape)
                # print('userpFedbayes_Y_shape:',Y.shape)
                batch_X = Variable(X.view(self.batch_size, -1))
                batch_Y = Variable(Y.view(self.batch_size, -1))
                # print('userpFedbayes_batch_X_shape:',batch_X.shape)
                # print('userpFedbayes_batch_Y_shape:',batch_Y.shape)
                ## local model
                # 1. Multiple sampling: sampling a group noise epsilons, its shape is equal to layer_param_shapes
                epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
                
                # 2. Reparameterization: assume the model parameter is guanssian distribution 
                layer_params1 = self.personal_model.transform_gaussian_samples(
                self.personal_model.mus, self.personal_model.rhos, epsilons)
                
                # 3. Calculating the prediction for each sample.
                personal_output= self.personal_model.net(batch_X, layer_params1)
                personal_outputs.append(personal_output.detach().cpu().numpy())
                
            prediction = np.mean(personal_outputs)
            uncertainty = np.std(personal_outputs)
            predictions_tensor = torch.tensor(prediction).float().to(self.device)
            uncertainties_tensor = torch.tensor(uncertainty).float().to(self.device)
            
            # 4. Variational Inference
            local_loss = self.personal_model.local_loss(
            predictions_tensor, uncertainties_tensor, batch_Y, layer_params1,
            self.personal_model.mus, self.personal_model.sigmas,
            copy.deepcopy(self.model.mus),
            [t.clone().detach() for t in self.model.sigmas], self.local_epochs)

            # 5. Stotistical Gradient Descent
            self.optimizer1.zero_grad()
            local_loss.backward()
            self.optimizer1.step()

            ## Global Model
            for m in range(1, N_Samples+1):
                # get all local models to update global parameters
                # 1. Multiple sampling
                epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
                
                # 2. Reparameterization:
                layer_params2 = self.model.transform_gaussian_samples(self.model.mus, self.model.rhos, epsilons)
                
                # 3. Calculating the prediction for each sample.
                output = self.model.net(batch_X, layer_params2)
                local_outputs.append(output.detach().cpu().numpy())
                
                # 4. Variational inference only KL divergence
                global_loss = self.model.global_loss(
                    [t.clone().detach() for t in layer_params1],
                    copy.deepcopy(self.personal_model.mus),
                    [t.clone().detach() for t in self.personal_model.sigmas],
                    self.model.mus, self.model.sigmas, self.local_epochs)

                # 5. Stotistical Gradient Descent
                self.optimizer2.zero_grad()
                global_loss.backward()
                self.optimizer2.step()

        return LOSS

