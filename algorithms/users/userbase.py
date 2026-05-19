import torch
from torch.nn import Module
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import numpy as np
import copy
from torch.autograd import Variable
from algorithms.trainmodel.models import *
import matplotlib.pyplot as plt
import pandas as pd

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
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        
        
        self.train_samples = sum([len(unit) for unit in train_data])
        self.test_samples = sum([len(unit) for unit in test_data])


        self.train_loaders = [] 
        if self.train_samples > 0:
            for unit_idx, unit_data in enumerate(train_data):
                if len(unit_data) > 0:
                    loader = DataLoader(unit_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
                    self.train_loaders.append(loader)
                else:
                    self.train_loaders.append(None)
        else:
            self.train_loaders = []


        self.test_loaders = []
        if self.test_samples > 0:
            for unit_idx, unit_data in enumerate(test_data):
                if len(unit_data) > 0:
                    loader = DataLoader(unit_data, batch_size=self.batch_size, shuffle=False)
                    self.test_loaders.append(loader)
                else:
                    self.test_loaders.append(None)
        else:
            self.test_loaders = []


        self.model = copy.deepcopy(model)   
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personal_model = copy.deepcopy(self.model)
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.device = device

        self.N_Batch = self.train_samples // batch_size
        data_dim = 30
        hidden_dim = 200
        total = (data_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (
                hidden_dim + 1) * 1
        L = 3
        a = np.log(total) + 0.1 * ((L + 1) * np.log(hidden_dim) + np.log(np.sqrt(self.train_samples) * data_dim))
        lm = 1 / np.exp(a)
        self.phi_prior = torch.tensor(lm).to(self.device)
        self.temp = 0.5

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()


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
        True_HI = []
        all_rmse = []
        all_mae = []
        all_epe = []
        preds_all = []
        y_all = []
        with torch.no_grad():
            for loader in self.test_loaders:
                if loader is None:
                    continue
                for x, y in loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)
                    pred = output.view(-1)      
                    y = y.view(-1)

                    pred_np = pred.cpu().numpy()
                    y_np = y.cpu().numpy()

                    rmse = np.sqrt(np.mean((pred_np - y_np) ** 2))
                    mae = np.mean(np.abs(pred_np - y_np))

                    epe = (np.mean(pred_np) - np.mean(y_np))**2 + np.var(pred_np)

                    all_rmse.append(rmse)
                    all_mae.append(mae)
                    all_epe.append(epe)

                    preds_all.append(pred_np)
                    y_all.append(y_np)
                self.update_parameters(self.local_model)

        if len(preds_all) > 0:
            predictions = np.concatenate(preds_all)
            True_HI = np.concatenate(y_all)
        else:
            predictions = np.array([])
            True_HI = np.array([])

            
        return all_rmse, all_mae, all_epe, len(predictions), predictions, True_HI
        
    def testBayes(self):
        self.model.eval()
        for x, y in self.testloaderfull:
            test_size = x.size()[0]
            test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
            test_Y = Variable(y.view(test_size, -1)).to(self.device)


            epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
            sigmas = self.model.transform_rhos(self.model.rhos)
            layer_params = self.model.transform_gaussian_samples(self.model.mus, sigmas, epsilons)
            output = self.model.net(test_X, layer_params)
            
            y = test_Y.data.view(test_size)
            rmse = np.sqrt(np.mean((output - y) ** 2))
            mae = np.mean(np.abs(output - y))

        return rmse, mae, y.shape[0]
    

    def visualize_all_units(self, glob_iter, preds_list, uncerts_list, trues_list, user_id):

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.unicode_minus'] = False

        num_units = len(trues_list)
        unit_idx = 0          
        num_units = 1
        if num_units == 0: 
            return
        
        rows = 1
        cols = 5
        if num_units > rows * cols:
            rows = (num_units + cols - 1) // cols


        figsize_height = max(6, rows * 3)
        fig, axes = plt.subplots(rows, cols, figsize=(30, figsize_height))
        if rows * cols > 1:
            axes = axes.flatten()
        else:

            axes = [axes]

        print(f"Visualizing {num_units} units for User {user_id} at Global Iter {glob_iter}...")

        for i in range(len(axes)):
            ax = axes[i]
            all_mae = []
            all_rmse = []
            if i < num_units:
                y_pred = preds_list[unit_idx].flatten()
                y_std  = uncerts_list[unit_idx].flatten()
                y_true = trues_list[unit_idx].flatten()
                unit_mae = np.mean(np.abs(y_pred - y_true))
                unit_rmse = np.sqrt(np.mean((y_pred - y_true)**2))

                print(
                        f"Unit {unit_idx} - "
                        f"Indicator_MAE: {unit_mae:.4f}, "
                        f"Indicator_RMSE: {unit_rmse:.4f}"
                    )
                
                try:
                    save_path = (
                        "Simulation_FedBayes/experimental_results/different_noise_results/"
                        "2noise-RUL_results_pFedBayes.csv"
                    )

                    row = {
                        "rmse": unit_rmse,
                        "mae": unit_mae,
                    }

                    df = pd.DataFrame([row])
                    write_header = not os.path.exists(save_path)

                    df.to_csv(
                        save_path,
                        mode="a",
                        header=write_header,
                        index=False
                    )
                except Exception as e:
                    print(f"Error saving results to CSV: {e}")


                all_mae.append(unit_mae)
                all_rmse.append(unit_rmse)
                x = np.linspace(0, 60, len(y_true))


                ax.fill_between(x, y_pred - 1.96*y_std, y_pred + 1.96*y_std,
                                color="#C7EDE6", alpha=0.4, label='95% Confidence Bands', zorder=1)

                ax.scatter(x, y_true, color='#A0A0A0', s=15, alpha=0.7, marker='o', label='Observed', zorder=2)

                ax.plot(x, y_pred, color='green', linewidth=2.5, linestyle='-', label='Fitted Value', zorder=3)

                ax.grid(False)
                ax.tick_params(axis='both', which='major', labelsize=16)
                
                if i == 0:
                    ax.legend(loc='best', frameon=True, shadow=False, fontsize=16)
                
                if i % cols == 0:
                    ax.set_ylabel(r"Degradation Signal $y(t)$", fontsize=16)

                if i >= (rows - 1) * cols or i + cols >= num_units:
                     ax.set_xlabel(r"Month $t$", fontsize=16)

            else:
                fig.delaxes(ax)
        
        plt.tight_layout()
        save_dir = "./Simulation_FedBayes/images"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"Global_Iter_{glob_iter}_User_{user_id}_All_Units_Prediction.pdf")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def testpFedbayes(self,glob_iter):
        self.personal_model.load_state_dict(self.model.state_dict())
        self.model.eval()
        self.personal_model.eval()
        N_samples = 50
        
        total_samples = 0  
        
        all_units_metrics = {
            "local_rmse": [], "local_mae": [], "local_epe": [],
            "global_rmse": [], "global_mae": [], "global_epe": [],
            "predictions": [], "uncertainties": [], "true_HI": [] 
        }

        for unit_idx, loader in enumerate(self.test_loaders):
            if loader is None: 
                continue 

            unit_preds_local = []
            unit_uncerts_local = []
            unit_preds_global = []
            unit_targets = []

            for batch_idx, (x, y) in enumerate(loader):
                test_size = x.size(0)
                total_samples += test_size 
                
                test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
                test_Y = Variable(y.view(test_size, -1).type(torch.FloatTensor)).to(self.device)

                batch_mc_means = []
                batch_mc_vars = []
                for m in range(N_samples):
                    epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
                    layer_params = self.personal_model.transform_gaussian_samples(
                        self.personal_model.mus, self.personal_model.rhos, epsilons)
                    mean, log_var = self.personal_model.net(test_X, layer_params)
                    batch_mc_means.append(mean.detach().cpu().numpy())
                    batch_mc_vars.append(torch.exp(log_var).detach().cpu().numpy())

                batch_pred_mean = np.mean(batch_mc_means, axis=0)
                epistemic_var = np.var(batch_mc_means, axis=0)
                aleatoric_var = np.mean(batch_mc_vars, axis=0)
                batch_pred_std = np.sqrt(epistemic_var + aleatoric_var)
                
                unit_preds_local.append(batch_pred_mean)
                unit_uncerts_local.append(batch_pred_std)

                batch_mc_means_g = []
                for m in range(N_samples):
                    epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
                    layer_params = self.model.transform_gaussian_samples(
                        self.model.mus, self.model.rhos, epsilons)
                    mean_g, _ = self.model.net(test_X, layer_params)
                    batch_mc_means_g.append(mean_g.detach().cpu().numpy())
                batch_pred_mean_g = np.mean(batch_mc_means_g, axis=0)
                unit_preds_global.append(batch_pred_mean_g)

                unit_targets.append(test_Y.detach().cpu().numpy())

            Y_true_flat = np.concatenate(unit_targets, axis=0)
            
            pred_local_flat = np.concatenate(unit_preds_local, axis=0)
            uncert_local_flat = np.concatenate(unit_uncerts_local, axis=0)
            pred_global_flat = np.concatenate(unit_preds_global, axis=0)

            Y_true_flat = Y_true_flat * 21.51194566480485
            pred_local_flat = pred_local_flat * 21.51194566480485
            uncert_local_flat = uncert_local_flat * 21.51194566480485
            pred_global_flat = pred_global_flat * 21.51194566480485

            local_rmse = np.sqrt(np.mean((pred_local_flat - Y_true_flat) ** 2))
            local_mae = np.mean(np.abs(pred_local_flat - Y_true_flat))
            local_epe = (np.mean(pred_local_flat) - np.mean(Y_true_flat))**2 + np.var(pred_local_flat)

            global_rmse = np.sqrt(np.mean((pred_global_flat - Y_true_flat) ** 2))
            global_mae = np.mean(np.abs(pred_global_flat - Y_true_flat))
            global_epe = (np.mean(pred_global_flat) - np.mean(Y_true_flat))**2 + np.var(pred_global_flat)

            all_units_metrics["local_rmse"].append(local_rmse)
            all_units_metrics["local_mae"].append(local_mae)
            all_units_metrics["local_epe"].append(local_epe)
            
            all_units_metrics["global_rmse"].append(global_rmse)
            all_units_metrics["global_mae"].append(global_mae)
            all_units_metrics["global_epe"].append(global_epe)
            
            all_units_metrics["predictions"].append(pred_local_flat)
            all_units_metrics["uncertainties"].append(uncert_local_flat)
            all_units_metrics["true_HI"].append(Y_true_flat)
        print("-------userbase_Y_true_flat_shape:",len(all_units_metrics["true_HI"]))

        num_valid_units = len(all_units_metrics["local_rmse"])
        
        if num_valid_units == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], 0

        avg_local_rmse = np.mean(all_units_metrics["local_rmse"])
        avg_local_mae  = np.mean(all_units_metrics["local_mae"])
        avg_local_epe  = np.mean(all_units_metrics["local_epe"])
        
        avg_global_rmse = np.mean(all_units_metrics["global_rmse"])
        avg_global_mae  = np.mean(all_units_metrics["global_mae"])
        avg_global_epe  = np.mean(all_units_metrics["global_epe"])

        self.visualize_all_units(glob_iter,
                                 all_units_metrics["predictions"],
                                 all_units_metrics["uncertainties"],
                                 all_units_metrics["true_HI"],
                                 user_id=self.id)

        return (avg_local_rmse, 
                avg_local_mae, 
                avg_local_epe, 
                avg_global_rmse, 
                avg_global_mae, 
                avg_global_epe,
                total_samples, 
                all_units_metrics["predictions"], 
                all_units_metrics["uncertainties"], 
                all_units_metrics["true_HI"],
                ) 

    def train_error_and_loss(self):
        self.model.eval()
        loss = 0
        all_rmse = []
        all_mae = []
        all_epe = []
        all_loss = []
        self.update_parameters(self.persionalized_model_bar)
        for loader in self.train_loaders:
            if loader is None: 
                continue
            
            for x, y in loader:
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
            return all_rmse, all_mae, all_epe, all_loss, len(self.train_loaders)


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
        self.personal_model.eval()
        total_local_mse = 0
        total_local_mae = 0
        total_local_epe = 0
        total_local_loss = 0
        total_global_mse = 0
        total_global_mae = 0
        total_global_epe = 0
        total_global_loss = 0
        
        total_samples = 0
        N_samples = 20 

        for loader in self.train_loaders:
            if loader is None: 
                continue
            
            for x, y in loader:
                batch_size = x.size(0)
                if x.ndim > 2:
                    train_X = Variable(x.view(batch_size, -1).type(torch.FloatTensor)).to(self.device)
                else:
                    train_X = Variable(x.type(torch.FloatTensor)).to(self.device)
                
                train_Y = Variable(y.view(batch_size, -1)).to(self.device)
                y_true_np = train_Y.cpu().detach().numpy() 
                batch_mc_means = []
                batch_mc_vars = []
                for m in range(N_samples):
                    epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
                    layer_params = self.personal_model.transform_gaussian_samples(
                        self.personal_model.mus, self.personal_model.rhos, epsilons)
                    
                    mean, log_var = self.personal_model.net(train_X, layer_params)
                    batch_mc_means.append(mean.detach().cpu().numpy())
                    batch_mc_vars.append(torch.exp(log_var).detach().cpu().numpy()) # 存 variance
                
                pred_mean = np.mean(batch_mc_means, axis=0)
                epistemic_var = np.var(batch_mc_means, axis=0)
                aleatoric_var = np.mean(batch_mc_vars, axis=0)
                total_var = epistemic_var + aleatoric_var
                pred_mean_tensor = torch.tensor(pred_mean).to(self.device)
                pred_log_var_tensor = torch.tensor(np.log(total_var + 1e-8), dtype=torch.float32).to(self.device)
                curr_local_loss = self.personal_model.local_loss(
                    pred_mean_tensor, pred_log_var_tensor, train_Y, layer_params,
                    self.personal_model.mus, self.personal_model.sigmas,
                    self.model.mus, self.model.sigmas, self.local_epochs
                ).item() 
                total_local_mse += np.sum((pred_mean - y_true_np) ** 2)
                total_local_mae += np.sum(np.abs(pred_mean - y_true_np))
                total_local_epe += np.sum((pred_mean - y_true_np)**2) + np.sum(total_var**2) # EPE公式优化
                total_local_loss += curr_local_loss * batch_size


                batch_mc_means_g = []
                batch_mc_vars_g = []
                for m in range(N_samples):
                    epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
                    layer_params_g = self.model.transform_gaussian_samples(
                        self.model.mus, self.model.rhos, epsilons)
                    mean_g, log_var_g = self.model.net(train_X, layer_params_g)
                    batch_mc_means_g.append(mean_g.detach().cpu().numpy())
                    batch_mc_vars_g.append(torch.exp(log_var_g).detach().cpu().numpy())
                    
                
                pred_mean_g = np.mean(batch_mc_means_g, axis=0)
                epistemic_var_g = np.var(batch_mc_means_g, axis=0)
                aleatoric_var_g = np.mean(batch_mc_vars_g, axis=0)
                total_var_g = epistemic_var_g + aleatoric_var_g # 总方差


                curr_global_loss = self.model.global_loss(
                    layer_params_g, 
                    self.personal_model.mus, self.personal_model.sigmas,
                    self.model.mus, self.model.sigmas, self.batch_size
                ).item()

                total_global_mse += np.sum((pred_mean_g - y_true_np) ** 2)
                total_global_mae += np.sum(np.abs(pred_mean_g - y_true_np))
                total_global_epe += np.sum((pred_mean_g - y_true_np)**2) + np.sum(total_var_g)
                total_global_loss += curr_global_loss * batch_size
                
                total_samples += batch_size

        # 4. 计算最终平均值 (Scalar)
        # Server 端期望的是单个数字，不是列表
        if total_samples == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        # RMSE = sqrt( Sum_SE / N )
        avg_local_rmse = np.sqrt(total_local_mse / total_samples)
        avg_local_mae = total_local_mae / total_samples
        avg_local_epe = total_local_epe / total_samples
        avg_local_loss = total_local_loss / total_samples
        
        avg_global_rmse = np.sqrt(total_global_mse / total_samples)
        avg_global_mae = total_global_mae / total_samples
        avg_global_epe = total_global_epe / total_samples
        avg_global_loss = total_global_loss / total_samples


        return (avg_local_rmse, avg_local_mae, avg_local_epe, 
                avg_global_rmse, avg_global_mae, avg_global_epe, 
                avg_local_loss, avg_global_loss, 
                total_samples, total_samples)

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
        self.personal_model.load_state_dict(self.model.state_dict())
        self.model.eval()

        all_rmse = []
        all_mae = []
        all_epe = []

        preds_all = []
        y_all = []
        self.update_parameters(self.persionalized_model_bar)

        with torch.no_grad():
            for loader in self.test_loaders:
                if loader is None:
                    continue

                for x, y in loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    output = self.model(x)     
                    pred = output.view(-1)     
                    y = y.view(-1)

                    pred_np = pred.cpu().numpy()
                    y_np = y.cpu().numpy()

                    # --- point-wise metrics ---
                    rmse = np.sqrt(np.mean((pred_np - y_np) ** 2))
                    mae = np.mean(np.abs(pred_np - y_np))

                    # --- EPE (distribution-level) ---
                    epe = (np.mean(pred_np) - np.mean(y_np))**2 + np.var(pred_np)

                    all_rmse.append(rmse)
                    all_mae.append(mae)
                    all_epe.append(epe)

                    preds_all.append(pred_np)
                    y_all.append(y_np)

        self.update_parameters(self.local_model)

        if len(preds_all) > 0:
            predictions = np.concatenate(preds_all)
            True_HI = np.concatenate(y_all)
        else:
            predictions = np.array([])
            True_HI = np.array([])

        return all_rmse, all_mae, all_epe, len(predictions), predictions, True_HI


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
        if not hasattr(self, 'iter_trainloader') or self.iter_trainloader is None:
            if hasattr(self, 'train_loaders') and len(self.train_loaders) > 0:
                self.iter_trainloader = iter(self.train_loaders[0])
            elif hasattr(self, 'trainloader'):
                self.iter_trainloader = iter(self.trainloader)
            else:
                return None, None 

        try:
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            if hasattr(self, 'train_loaders') and len(self.train_loaders) > 0:
                self.iter_trainloader = iter(self.train_loaders[0])
            else:
                self.iter_trainloader = iter(self.trainloader)
            
            (X, y) = next(self.iter_trainloader)

        return X, y    

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
