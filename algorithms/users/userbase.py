import torch
from torch.nn import Module
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import numpy as np
import copy
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from algorithms.trainmodel.models import *


class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, K, id, train_data, test_data, model, batch_size=0, learning_rate=0, beta=0, lamda=0,
                 local_epochs=0, device=torch.device('cpu'), output_dim=10):
        # from fedprox
        self.output_dim = output_dim
        self.model = copy.deepcopy(model) if isinstance(model, Module) else model().to(device)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size, drop_last=True)
        self.testloader = DataLoader(test_data, self.test_samples, drop_last=True)
        self.testloaderfull = DataLoader(test_data, self.test_samples, drop_last=True)
        self.trainloaderfull = DataLoader(train_data, self.batch_size, drop_last=True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personal_model = copy.deepcopy(model)
        # self.local_model = copy.deepcopy(model)
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.device = device

        # with torch.no_grad():
        #     self.personal_model.weight.fill_(model.weight)
        #     self.model.weight.fill_(model.weight)

        self.N_Batch = len(train_data) // batch_size
        self.data_size = len(train_data)
        data_dim = 30
        hidden_dim = 200
        total = (data_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (
                hidden_dim + 1) * 1
        L = 3
        a = np.log(total) + 0.1 * ((L + 1) * np.log(hidden_dim) + np.log(np.sqrt(self.data_size) * data_dim))
        lm = 1 / np.exp(a)
        self.phi_prior = torch.tensor(lm).to(self.device)
        self.temp = 0.5

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        # self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def set_parameters_pFed(self, model):
        for user_layer, server_layer in zip(self.model.layers, model.layers):
            for personal_param, local_param, new_param in zip(user_layer.personal.parameters(),
                                                              user_layer.local.parameters(),
                                                              server_layer.local.parameters()):
                personal_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        predictions = []
        uncertainties = []
        True_HI = []
        all_rmse = []
        all_mae = []
        all_epe = []
        for x, y in self.testloaderfull:
            output = self.model(x)
            y = y.cpu().detach().numpy()
            prediction = output.cpu().detach().numpy()
            rmse = np.sqrt(np.mean((prediction - y) ** 2))
            mae = np.mean(np.abs(prediction - y))
            epe = (np.mean(prediction) - np.mean(y))**2 + np.var(prediction)
            predictions.append(prediction)
            uncertainties.append(0)
            True_HI.append(y)
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_epe.append(epe)
            
        return all_rmse, all_mae, all_epe, len(self.testloader), predictions, uncertainties, True_HI
        
    def testBayes(self):
        self.model.eval()
        for x, y in self.testloaderfull:
            test_size = x.size()[0]
            test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
            test_Y = Variable(y.view(test_size, -1)).to(self.device)
            # output = self.model.forward(test_X, mode='MAP').data.argmax(axis=1)

            epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
            # compute softplus for variance
            sigmas = self.model.transform_rhos(self.model.rhos)
            # obtain a sample from q(w|theta) by transforming the epsilons
            layer_params = self.model.transform_gaussian_samples(self.model.mus, sigmas, epsilons)
            # forward-propagate the batch
            output = self.model.net(test_X, layer_params)
            
            # output = F.softmax(output, dim=1).data.argmax(axis=1)
            y = test_Y.data.view(test_size)
            rmse = np.sqrt(np.mean((output - y) ** 2))
            mae = np.mean(np.abs(output - y))

        return rmse, mae, y.shape[0]

    def testpFedbayes(self):
        self.model.eval()
        N_samples = 20
        True_HI = []
        outputs = []
        global_outputs = []
        predictions = []
        uncertainties = []
        global_predictions = []
        global_uncertainties = []
        all_global_rmse = []
        all_global_mae = []
        all_global_epe = []
        all_local_rmse = []
        all_local_mae = []
        all_local_epe  = []
        # print("userbase_testloader_len:", len(self.testloaderfull))
        for x, y in self.testloaderfull:
            for m in range(1, N_samples+1):
                test_size = x.size()[0]
                test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
                # print('userbase_test_X_shape:',test_X.shape)
                test_Y = Variable(y.view(test_size, -1)).to(self.device)
                
                # print('userbase_test_Y_shape:',test_Y.shape)
                # personal model
                epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
                # obtain a sample from q(w|theta) by transforming the epsilons
                layer_params1 = self.personal_model.transform_gaussian_samples(self.personal_model.mus,
                                                                            self.personal_model.rhos, epsilons)
                # forward-propagate the batch
                output = self.personal_model.net(test_X, layer_params1)
                outputs.append(output.detach().cpu().numpy())
                # print('userbase_output_shape:',output.shape)
            local_prediction = np.mean(outputs, axis=0)
            local_uncertainty = np.std(outputs, axis=0)
            predictions.append(local_prediction)
            uncertainties.append(local_uncertainty)
                
            test_Y_local = test_Y.cpu().detach().numpy()
            True_HI.append(test_Y_local)
            
            local_rmse = np.sqrt(np.mean((predictions - test_Y_local) ** 2))
            local_mae = np.mean(np.abs(predictions - test_Y_local))
            local_epe = (np.mean(predictions) - np.mean(test_Y_local))**2 + np.var(predictions)
            all_local_rmse.append(local_rmse)
            all_local_mae.append(local_mae)
            all_local_epe.append(local_epe)

            for m in range(1, N_samples+1):
                # global model
                epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
                # obtain a sample from q(w|theta) by transforming the epsilons
                layer_params1 = self.model.transform_gaussian_samples(self.model.mus, self.model.rhos, epsilons)
                # forward-propagate the batch
                global_output = self.model.net(test_X, layer_params1)
                global_outputs.append(global_output.detach().cpu().numpy())
            global_prediction = np.mean(global_outputs, axis=0)
            global_uncertainty = np.std(global_outputs, axis=0)
            global_predictions.append(global_prediction)
            global_uncertainties.append(global_uncertainty)
                
            test_Y_gloal = test_Y.cpu().detach().numpy()
            global_rmse = np.sqrt(np.mean((global_predictions - test_Y_gloal) ** 2))
            global_mae = np.mean(np.abs(global_predictions - test_Y_gloal))
            global_epe = (np.mean(global_predictions) - np.mean(test_Y_gloal))**2 + np.var(global_predictions)
            all_global_rmse.append(global_rmse)
            all_global_mae.append(global_mae)
            all_global_epe.append(global_epe)
           
        return all_local_rmse, all_local_mae, all_local_epe,  all_global_rmse, all_global_mae, all_global_epe, len(self.testloaderfull), predictions, uncertainties, True_HI

    def testSparseBayes(self):
        # self.model.eval()
        for x, y in self.testloaderfull:
            test_size = x.size()[0]
            test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
            test_Y = Variable(y.view(test_size, -1)).to(self.device)
            output = self.model.forward(test_X, mode='MAP').data.argmax(axis=1)
            loss, pred = self.model.sample_elbo(test_X, test_Y, 30, self.temp, self.phi_prior, self.N_Batch)
            pred = pred.mean(dim=0)
            output = pred.data.argmax(axis=1)
            y = test_Y.data.view(test_size)
            rmse = np.sqrt(np.mean((output - y) ** 2))
            mae = np.mean(np.abs(output - y))

        return rmse, mae, y.shape[0]

    def testpFedSbayes(self):
        # self.model.eval()
        for x, y in self.testloaderfull:
            test_size = x.size()[0]
            test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
            test_Y = Variable(y.view(test_size, -1)).to(self.device)
            # output = self.model.forward(test_X, mode='MAP').data.argmax(axis=1)
            loss, pred = self.model.sample_elbo(test_X, test_Y, 30, self.temp, self.phi_prior, self.N_Batch)
            pred = pred.mean(dim=0)
            output = pred.data.argmax(axis=1)
            y = test_Y.data.view(test_size)
            rmse = np.sqrt(np.mean((output - y) ** 2))
            rmse.append(rmse)
            mae = np.mean(np.abs(output - y))
        avg_rmse = np.sum(rmse) / len(test_Y)
        return avg_rmse, mae, y.shape[0]

    def train_error_and_loss(self):
        self.model.eval()
        loss = 0
        all_rmse = []
        all_mae = []
        all_epe = []
        all_loss = []
        for x, y in self.trainloaderfull:
            output = self.model(x)
            y = y.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            rmse = np.sqrt(np.mean((output - y) ** 2))
            mae = np.mean(np.abs(output - y))
            epe = (np.mean(output) - np.mean(y))**2 + np.var(output)
            loss = self.loss(torch.tensor(output), torch.tensor(y)).item()
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_epe.append(epe)
            all_loss.append(loss)
    
        return all_rmse, all_mae, all_epe, all_loss, len(self.trainloaderfull)

    def train_error_and_loss_bayes(self):
        self.model.eval()
        loss = 0
        for x, y in self.trainloaderfull:
            size = x.size()[0]
            train_X = Variable(x.view(size, -1).type(torch.FloatTensor)).to(self.device)
            train_Y = Variable(y.view(size, -1)).to(self.device)

            label_one_hot = F.one_hot(train_Y, num_classes=self.output_dim).squeeze(dim=1)
            epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
            # compute softplus for variance
            sigmas = self.model.transform_rhos(self.model.rhos)
            # obtain a sample from q(w|theta) by transforming the epsilons
            layer_params = self.model.transform_gaussian_samples(self.model.mus, sigmas, epsilons)
            # forward-propagate the batch
            output = self.model.net(train_X, layer_params)
            # calculate the loss
            loss = self.model.combined_loss(output, label_one_hot, layer_params, self.model.mus, sigmas,
                                            self.local_epochs)

            # output = F.softmax(output, dim=1).data.argmax(axis=1)
            y = train_Y.data.view(size)
            rmse = np.sqrt(np.mean((output - y) ** 2))
            mae = np.mean(np.abs(output - y))
            
        return rmse, mae, loss, self.train_samples

    def train_error_and_loss_pFedbayes(self):
        self.model.eval()
        local_outputs = []
        global_outputs = []
        local_total_loss = []
        global_total_loss = []
        local_total_samples = []
        global_total_samples = []
        N_samples = 20
        all_local_rmse = []
        all_local_mae = []
        all_local_epe = []
        all_global_rmse = []
        all_global_mae = []
        all_global_epe = []
        
        # print("user_base_self.trainloader_len: ", len(self.trainloader))
        for x, y in self.trainloader:
            for m in range(1, N_samples+1):
                size = x.size()[0]
                train_X = Variable(x.view(size, -1).type(torch.FloatTensor)).to(self.device)
                train_Y = Variable(y.view(size, -1)).to(self.device)
                ### personal model
                epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
                # obtain a sample from q(w|theta) by transforming the epsilons
                layer_params1 = self.personal_model.transform_gaussian_samples(self.personal_model.mus,self.personal_model.rhos, epsilons)
                # forward-propagate the batch
                local_output = self.personal_model.net(train_X, layer_params1)
                local_outputs.append(local_output.detach().cpu().numpy())
            prediction = np.mean(local_outputs, axis=0)
            uncertainty = np.std(local_outputs, axis=0)

            predictions_tensor = torch.tensor(prediction).to(self.device)  # 转换为张量
            uncertainties_tensor = torch.tensor(uncertainty).to(self.device)  # 转换为张量
            # calculate the local loss
            local_loss = self.personal_model.local_loss(predictions_tensor, uncertainties_tensor, train_Y, layer_params1,
                                                              self.personal_model.mus, self.personal_model.sigmas,
                                                              self.model.mus, self.model.sigmas, self.local_epochs)
            predictions_tensor = predictions_tensor.cpu().detach().numpy()
            train_Y = train_Y.cpu().detach().numpy()
            local_rmse = np.sqrt(np.mean((predictions_tensor - train_Y) ** 2))
            local_mae = np.mean(np.abs(predictions_tensor - train_Y))
            local_epe = (np.mean(predictions_tensor) - np.mean(train_Y))**2 + np.var(predictions_tensor)
            all_local_rmse.append(local_rmse)
            all_local_mae.append(local_mae)
            all_local_epe.append(local_epe)
            local_total_loss.append(local_loss)
            local_total_samples = train_Y.shape[0]
            
            for m in range(1, N_samples+1):
                epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
                # obtain a sample from q(w|theta) by transforming the epsilons
                layer_params1 = self.model.transform_gaussian_samples(self.personal_model.mus,self.personal_model.rhos, epsilons)
                # forward-propagate the batch
                global_output = self.model.net(train_X, layer_params1)
                global_outputs.append(global_output.detach().cpu().numpy())
            global_prediction = np.mean(global_outputs, axis=0)
            global_uncertainty = np.std(global_outputs, axis=0)

            global_predictions_tensor = torch.tensor(global_prediction).to(self.device)  # 转换为张量
            global_uncertainties_tensor = torch.tensor(global_uncertainty).to(self.device)  # 转换为张量
            # calculate the local loss
            global_loss = self.personal_model.global_loss(layer_params1,
                                                              self.personal_model.mus, self.personal_model.sigmas,
                                                              self.model.mus, self.model.sigmas, self.batch_size)
            global_predictions_tensor = global_predictions_tensor.cpu().detach().numpy()
            # train_Y = train_Y.cpu().detach().numpy()
            global_rmse = np.sqrt(np.mean((global_predictions_tensor - train_Y) ** 2))
            global_mae = np.mean(np.abs(global_predictions_tensor - train_Y))
            global_epe = (np.mean(global_predictions_tensor) - np.mean(train_Y))**2 + np.var(global_predictions_tensor)
            all_global_rmse.append(global_rmse)
            all_global_epe.append(global_epe)
            all_global_mae.append(global_mae)
            global_total_loss.append(global_loss)
            global_total_samples = train_Y.shape[0]
        
        return all_local_rmse, all_local_mae, all_local_epe, all_global_rmse, all_global_mae, all_global_epe, local_total_loss, global_total_loss, len(self.trainloader), len(self.trainloader)

    def train_error_and_loss_sparsebayes(self):
        self.model.eval()
        loss = 0
        for x, y in self.trainloaderfull:
            size = x.size()[0]
            train_X = Variable(x.view(size, -1).type(torch.FloatTensor)).to(self.device)
            train_Y = Variable(y.view(size, -1)).to(self.device)
            output = self.model.forward(train_X, mode='MAP').data.argmax(axis=1)
            y = train_Y.data.view(size)
            rmse = np.sqrt(np.mean((output - y) ** 2))
            mae = np.mean(np.abs(output - y))

            loss += self.loss.loss_fn(output, y, 1.0)

        return rmse, mae, loss, self.train_samples

    def train_error_and_loss_pFedSbayes(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            size = x.size()[0]
            train_X = Variable(x.view(size, -1).type(torch.FloatTensor)).to(self.device)
            train_Y = Variable(y.view(size, -1)).to(self.device)
            loss_temp, pred = self.model.sample_elbo(train_X, train_Y, 30, self.temp, self.phi_prior, self.N_Batch)
            pred = pred.mean(dim=0)
            output = pred.data.argmax(axis=1)
            y = train_Y.data.view(size)
            rmse = np.sqrt(np.mean((output - y) ** 2))
            mae = np.mean(np.abs(output - y))
            loss += loss_temp

        return rmse, mae, loss, self.train_samples

    def test_persionalized_model(self):
        self.model.eval()
        predictions = []
        uncertainties = []
        True_HI = []
        all_rmse = []
        all_mae = []
        all_epe = []
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.testloaderfull:
            output = self.model(x)
            y = y.cpu().detach().numpy()
            prediction = output.cpu().detach().numpy()
            rmse = np.sqrt(np.mean((prediction - y) ** 2))
            mae = np.mean(np.abs(prediction - y))
            epe = (np.mean(prediction) - np.mean(y))**2 + np.var(prediction)
            predictions.append(prediction)
            uncertainties.append(0)
            True_HI.append(y)
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_epe.append(epe)

        self.update_parameters(self.local_model)

        return all_rmse, all_mae, all_epe, len(self.testloaderfull), predictions, uncertainties, True_HI

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        loss = 0
        all_rmse = []
        all_mae = []
        all_epe = []
        all_loss = []
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.trainloader:
            output = self.model(x)
            y = y.cpu().detach().numpy()
            prediction = output.cpu().detach().numpy()
            rmse = np.sqrt(np.mean((prediction - y) ** 2))
            mae = np.mean(np.abs(prediction - y))
            epe = (np.mean(prediction) - np.mean(y))**2 + np.var(prediction)
            loss = self.loss(torch.tensor(prediction), torch.tensor(y)).item()
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_loss.append(loss)
            all_epe.append(epe)

        self.update_parameters(self.local_model)
        return all_rmse, all_mae, all_epe, all_loss, len(self.trainloaderfull)

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X, y)

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
