import torch
import os
import h5py
import copy
from torch.nn import Module
import numpy as np
from scipy.integrate import quad
from math import exp
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

#----------------------------pFedBayes，pFedSBayes，Our-No-Per----------------------------------
def quick_rul_calculator(time_arr, true_G_list, pred_G_list, w_val):
    # 1. 扁平化数据
    def to_flat_numpy(x):
        if x is None: return np.array([])
        if isinstance(x, torch.Tensor): x = x.cpu().detach().numpy()
        if isinstance(x, (list, tuple)): x = np.array(x)
        return x.reshape(-1)

    true_G_list = to_flat_numpy(true_G_list)
    pred_G_list = to_flat_numpy(pred_G_list)
    time_arr = to_flat_numpy(time_arr)


    def solve_any_rul(data_list, t_list, label="Data"):
        MAX_RUL_LIMIT = 25.0  

        if len(data_list) == 0: return 0.0
        curr_val = data_list[-1]
        if len(t_list) == len(data_list):
            curr_t = t_list[-1]
            start_t = t_list[0]
        else:
            curr_t = float(len(data_list))
            start_t = 0.0

        if len(data_list) < 2:
            slope = (curr_val - 0.0) / (curr_t - 0.0 + 1e-9)
        else:
            slope = (curr_val - data_list[0]) / (curr_t - start_t + 1e-9)

        THRESHOLD = 20.0

        if slope < 0:
            target = -THRESHOLD
            if curr_val <= target: return 0.0
            dist = curr_val - target
            speed = abs(slope)
        else:
            target = THRESHOLD
            if curr_val >= target: return 0.0
            dist = target - curr_val
            speed = slope

        if speed < 1e-5: speed = 1e-5 
        rul = dist / speed
        
        if rul > MAX_RUL_LIMIT:
            rul = MAX_RUL_LIMIT
            
        return rul

    r_t = solve_any_rul(true_G_list, time_arr, "TrueG")
    r_p = solve_any_rul(pred_G_list, time_arr, "PredG")

    return r_t, r_p


class Server:
    def __init__(self, K, dataset, algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, device,
                 local_learning_rate, num_units, scenario, alpha):

        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.total_test_samples = 0
        self.model = copy.deepcopy(model) if isinstance(model, Module) else model().to(device)  # global model.
        self.train_users = []
        self.test_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_global_rmse, self.rs_train_local_rmse, self.rs_test_global_rmse, self.rs_test_local_rmse, self.rs_train_global_loss, self.rs_train_local_loss = [], [], [], [], [], []
        self.times = times

 #-------------------the way of aggregating gradients---------------------       
    def aggregate_grads(self): 
        assert (self.train_users is not None and len(self.train_users) > 0)  
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.train_users:
            self.add_grad(user, user.train_samples / self.total_train_samples)
    def add_grad(self, user, ratio): 
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio
    def send_parameters(self):
        assert (self.train_users is not None and len(self.train_users) > 0)
        for user in self.train_users:
            user.set_parameters(self.model)

        if self.test_users is not None and len(self.test_users) > 0:
            for user in self.test_users:
                user.set_parameters(self.model)
    
#-------------------the way of aggregating parameters--------------------- 
    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.train_users is not None and len(self.train_users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.train_users:
            total_train += user.train_samples
        for user in self.train_users:
            self.add_parameters(user, user.train_samples / total_train)
    def save_model(self, post_fix_str):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_name = os.path.join(model_path, "server_" + post_fix_str + ".pt")
        torch.save(self.model, file_name)
        return file_name
    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)
    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.train_users)):
            return self.train_users

        num_users = min(num_users, len(self.train_users))
        
        return self.train_users

    def persionalized_update_parameters(self,user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def persionalized_aggregate_parameters(self):
        assert (self.train_users is not None and len(self.train_users) > 0)

        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0

        for user in self.train_users:
            total_train += user.train_samples

        for user in self.train_users:
            self.add_parameters(user, user.train_samples / total_train)

        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    def save_results(self, post_fix_str):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)

        alg = alg + "_" + str(self.times)


        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)

        alg = alg + "_" + str(self.times)
        if (len(self.rs_test_local_rmse) != 0 & len(self.rs_train_local_rmse) & len(self.rs_train_local_loss)) :
            with h5py.File("/home/ovo/WZJ/3. Federated_Bayesian_for_RUL/Simulation_FedBayes/results/"+'{}.h5'.format(alg + '_' + post_fix_str, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_per_rmse', data=self.rs_test_local_rmse)
                hf.create_dataset('rs_glob_rmse', data=self.rs_test_global_rmse)
                hf.create_dataset('rs_train_rmse', data=self.rs_train_local_rmse)
                hf.create_dataset('rs_train_loss', data=self.rs_train_local_loss)
                hf.close()


    def test(self):
        if not hasattr(self, "test_users") or len(self.test_users) == 0:
            return np.array([]), np.array([]), np.array([]), 0, np.array([]), np.array([]), np.array([])

        if not hasattr(self, "_round_in_time"):
            self._round_in_time = 0
        if not hasattr(self, "_time_id"):
            self._time_id = 0

        self._round_in_time += 1

        true_ruls, pred_ruls = [], []
        g_trues, g_preds = [], []
        for c in self.test_users:
            stats = c.test()
            if stats is None:
                continue

            try:
                _, _, _, _, Pre_mean_HI, True_HI = stats
            except Exception:
                Pre_mean_HI = stats[4]
                True_HI = stats[-1]

            if True_HI is None or len(True_HI) == 0:
                continue

            for i in range(len(True_HI)):
                g_true = np.asarray(True_HI[i]).reshape(-1)
                g_pred = np.asarray(Pre_mean_HI[i]).reshape(-1)

                fixed_time = np.linspace(
                    0, 60, max(len(g_true), len(g_pred), 2)
                )

                r_true, r_pred = quick_rul_calculator(
                    fixed_time, g_true, g_pred, w_val=1
                )

                true_ruls.append(r_true)
                pred_ruls.append(r_pred)

                g_trues.append(g_true)
                g_preds.append(g_pred) 

        if len(true_ruls) == 0:
            return np.array([]), np.array([]), np.array([]), 0, np.array([]), np.array([]), np.array([])

        rmse = np.sqrt(mean_squared_error(true_ruls, pred_ruls))
        mae = mean_absolute_error(true_ruls, pred_ruls)
        epe = (np.mean(true_ruls) - np.mean(pred_ruls)) ** 2 + np.var(pred_ruls)


        if self._round_in_time < self.times:
            return (
                np.array([rmse]),
                np.array([mae]),
                np.array([epe]),
                0, np.array([]), np.array([]), np.array([])
            )

        self._time_id += 1
        self._round_in_time = 0   


        try:
            save_path = (
                "./"
                "Simulation_FedBayes/experimental_results/pFedDNN_Scenario_I/"
                "RUL_results_FedDNN.csv"
            )

            row = {
                "time_id": self._time_id,
                "rmse": rmse,
                "mae": mae,
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
            print("[Warning] Failed to save CSV:", e)

        return (
            np.array([rmse]),
            np.array([mae]),
            np.array([epe]),
            0, np.array([]), np.array([]), np.array([])
        )


    def testBayes(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_rmse = []
        tot_mae = []

        for c in self.test_users:
            c_rmse, c_mae, ns = c.testBayes()
            tot_rmse.append(c_rmse*1.0)
            tot_mae.append(c_mae*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.test_users]

        return ids, num_samples, tot_rmse, tot_mae

    def testpFedbayes(self, glob_iter):
        if not hasattr(self, "_round_in_time_pfed"):
            self._round_in_time_pfed = 0
        if not hasattr(self, "_time_id_pfed"):
            self._time_id_pfed = 0

        self._round_in_time_pfed += 1
        true_ruls = []
        pred_ruls = []

        for c in self.test_users:
            (
                local_rmse, local_mae, local_epe,
                global_rmse, global_mae, global_epe,
                ns, Pre_mean_HI, Pre_std_HI, True_HI
            ) = c.testpFedbayes(glob_iter)

            for i in range(len(True_HI)):
                g_true = True_HI[i].flatten()
                g_pred = Pre_mean_HI[i].flatten()

                fixed_time = np.linspace(0, 60, 600)

                r_true, r_pred = quick_rul_calculator(
                    fixed_time, g_true, g_pred, w_val=1
                )

                true_ruls.append(r_true)
                pred_ruls.append(r_pred)


        if len(true_ruls) == 0:
            return None, None

        MAE = mean_absolute_error(true_ruls, pred_ruls)
        RMSE = np.sqrt(mean_squared_error(true_ruls, pred_ruls))

        print("-" * 30)
        print("RUL RMSE:", RMSE)
        print("RUL MAE:", MAE)

        if self._round_in_time_pfed < self.times:
            return RMSE, MAE


        self._time_id_pfed += 1
        self._round_in_time_pfed = 0  


        try:
            save_dir = (
                "Simulation_FedBayes/experimental_results/different_noise_results/"
            )
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir, "RUL_results_pFedBayes.csv"
            )

            row = {
                "time_id": self._time_id_pfed,
                "glob_iter": glob_iter,
                "rmse": RMSE,
                "mae": MAE,
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
            print("[Warning] Failed to save pFedBayes CSV:", e)

        return RMSE, MAE


    def testSparseBayes(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_rmse = []
        tot_mae = []
        for c in self.test_users:
            c_rmse, c_mae, ns = c.testSparseBayes()
            tot_rmse.append(c_rmse*1.0)
            tot_mae.append(c_mae*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.test_users]

        return ids, num_samples, tot_rmse, tot_mae



    def train_error_and_loss(self):
        tot_local_samples = []
        tot_global_samples = []
        tot_local_rmse = []
        tot_local_mae = []
        tot_local_epe = []
        tot_global_rmse = []
        tot_global_mae = []
        tot_global_epe = []
        local_losses = []
        global_losses = []


        for c in self.train_users:
            stats = c.train_error_and_loss()
            
            (local_rmse, local_mae, local_epe, 
             global_rmse, global_mae, global_epe, 
             c_local_loss, c_global_loss, 
             c_local_samples, c_global_samples) = stats


            if c_local_samples == 0:
                continue


            tot_local_rmse.append(local_rmse)
            tot_local_mae.append(local_mae)
            tot_local_epe.append(local_epe)
            
            tot_global_rmse.append(global_rmse)
            tot_global_mae.append(global_mae)
            tot_global_epe.append(global_epe)
            
            tot_local_samples.append(c_local_samples)
            tot_global_samples.append(c_global_samples)
            
            local_losses.append(c_local_loss)
            global_losses.append(c_global_loss)

        ids = [c.id for c in self.train_users]

        return tot_local_rmse, tot_local_mae, tot_local_epe, \
               tot_global_rmse, tot_global_mae, tot_global_epe, \
               local_losses, global_losses, \
               tot_local_samples, tot_global_samples

    def train_error_and_loss_bayes(self):
        num_samples = []
        tot_rmse = []
        tot_mae = []
        losses = []
        for c in self.train_users:
            c_rmse, c_mae, c_loss, ns = c.train_error_and_loss_bayes()
            tot_rmse.append(c_rmse * 1.0)
            tot_mae.append(c_mae * 1.0)
            num_samples.append(ns)
            losses.append(c_loss * 1.0)

        ids = [c.id for c in self.train_users]
 

        return ids, num_samples, tot_rmse, tot_mae, losses

    def train_error_and_loss_pFedbayes(self):
        tot_local_samples = []
        tot_global_samples = []
        tot_local_rmse = []
        tot_local_mae = []
        tot_local_epe = []
        tot_global_rmse = []
        tot_global_mae = []
        tot_global_epe = []
        local_losses = []
        global_losses = []

        for c in self.train_users:
            stats = c.train_error_and_loss_pFedbayes()
            (local_rmse, local_mae, local_epe, 
             global_rmse, global_mae, global_epe, 
             c_local_loss, c_global_loss, 
             c_local_samples, c_global_samples) = stats

            if c_local_samples == 0:
                continue

            tot_local_rmse.append(local_rmse)
            tot_local_mae.append(local_mae)
            tot_local_epe.append(local_epe)
            
            tot_global_rmse.append(global_rmse)
            tot_global_mae.append(global_mae)
            tot_global_epe.append(global_epe)
            
            tot_local_samples.append(c_local_samples)
            tot_global_samples.append(c_global_samples)
            
            local_losses.append(c_local_loss)
            global_losses.append(c_global_loss)

        ids = [c.id for c in self.train_users]

        return tot_local_rmse, tot_local_mae, tot_local_epe, \
               tot_global_rmse, tot_global_mae, tot_global_epe, \
               local_losses, global_losses, \
               tot_local_samples, tot_global_samples
    
    def train_error_and_loss_sparsebayes(self):
        num_samples = []
        tot_rmse = []
        tot_mae = []
        losses = []

        for c in self.train_users:
            c_rmse, c_mae, c_loss, ns = c.train_error_and_loss_sparsebayes()
            tot_rmse.append(c_rmse * 1.0)
            tot_mae.append(c_mae * 1.0)
            num_samples.append(ns)
            losses.append(c_loss * 1.0)

        ids = [c.id for c in self.train_users]

        
        return ids, num_samples, tot_rmse, tot_mae, losses

    def train_error_and_loss_pFedSbayes(self):
        num_samples = []
        tot_rmse = []
        tot_mae = []
        losses = []
        for c in self.train_users:
            c_rmse, c_mae, c_loss, ns = c.train_error_and_loss_sparsebayes()
            tot_rmse.append(c_rmse * 1.0)
            tot_mae.append(c_mae * 1.0)
            num_samples.append(ns)
            losses.append(c_loss * 1.0)

        ids = [c.id for c in self.train_users]


        return ids, num_samples, tot_rmse, tot_mae, losses



    def test_persionalized_model(self):
        if not hasattr(self, "_round_in_time_pfed"):
            self._round_in_time_pfed = 0
        if not hasattr(self, "_time_id_pfed"):
            self._time_id_pfed = 0

        self._round_in_time_pfed += 1
        true_ruls = []
        pred_ruls = []    
        for c in self.test_users:
            local_rmse, local_mae, local_epe, ns, Pre_mean_HI, True_HI = c.test_persionalized_model()
            for i in range(len(True_HI)):
                g_true = True_HI[i].flatten()
                g_pred = Pre_mean_HI[i].flatten()
                fixed_time = np.linspace(0, 60, 600)
                r_true, r_pred = quick_rul_calculator(fixed_time, g_true, g_pred, w_val=1)
                true_ruls.append(r_true)
                pred_ruls.append(r_pred)

        if len(true_ruls) == 0:
            return None, None

        MAE = mean_absolute_error(true_ruls, pred_ruls)
        RMSE = np.sqrt(mean_squared_error(true_ruls, pred_ruls))

        print("-" * 30)
        print("RUL RMSE:", RMSE)
        print("RUL MAE:", MAE)


        if self._round_in_time_pfed < self.times:
            return RMSE, MAE

        self._time_id_pfed += 1
        self._round_in_time_pfed = 0  

        try:
            save_dir = (
                "Simulation_FedBayes/experimental_results/pFedME_Scenario_I/"
            )
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir, "RUL_results_pFedME.csv"
            )

            row = {
                "time_id": self._time_id_pfed,
                "rmse": RMSE,
                "mae": MAE,
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
            print("[Warning] Failed to save pFedBayes CSV:", e)

        return RMSE, MAE
    
    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_rmse = []
        tot_mae = []
        tot_epe = []
        losses = []

        for c in self.train_users:
            local_rmse, local_mae, local_epe, c_loss, ns = c.train_error_and_loss_persionalized_model() 
            tot_rmse.append(local_rmse)
            tot_mae.append(local_mae)
            tot_epe.append(local_epe)
            num_samples.append(ns)
            losses.append(c_loss)
        
        ids = [c.id for c in self.train_users]

        return ids, num_samples, tot_rmse, tot_mae, tot_epe, losses
    

    def evaluate(self):
        Average_Test_Local_RMSE = []
        Average_Test_Local_MAE = []
        stats_test = self.test()
        stats_train = self.train_error_and_loss()
        
        test_local_rmse = np.array([sum(x) for x in stats_test[2]]) / stats_test[1]
        test_local_mae = np.array([sum(x) for x in stats_test[3]]) / stats_test[1]
        test_local_epe = np.array([sum(x) for x in stats_test[4]]) / stats_test[1]
        
        train_local_rmse = np.array([sum(x) for x in stats_train[2]]) / stats_train[1]
        train_local_mae = np.array([sum(x) for x in stats_train[3]]) / stats_train[1]
        train_local_epe = np.array([sum(x) for x in stats_train[4]]) / stats_train[1]
        self.rs_test_global_rmse.append(test_local_rmse)
        self.rs_train_local_rmse.append(train_local_rmse)

        
        print("Average Test Local RMSE: ", test_local_rmse)
        print("Average Test Local MAE: ", test_local_mae)
        Average_Test_Local_RMSE.append(test_local_rmse)
        Average_Test_Local_MAE.append(test_local_mae)
        
        return stats_test[6], stats_test[7], stats_test[8]
          
    def evaluate_pFedbayes(self, glob_iter):
        stats_test = self.testpFedbayes(glob_iter) 
        test_local_rmse = np.mean(stats_test[0])
        test_local_mae = np.mean(stats_test[1])
        self.rs_test_local_rmse.append(test_local_rmse)

        print(" ")
        print("Evaluate_Local_Model")
        print("Average Test local  RMSE: ", test_local_rmse)
        print("Average Test local  MAE: ", test_local_mae)

        return test_local_rmse, test_local_mae  

    def evaluate_personalized_model(self):
        stats_test = self.test_persionalized_model()  
        test_local_rmse = np.mean(stats_test[0])
        test_local_mae = np.mean(stats_test[1])
        self.rs_test_local_rmse.append(test_local_rmse)

        print(" ")
        print("Evaluate_Local_Model")
        print("Average Test local  RMSE: ", test_local_rmse)
        print("Average Test local  MAE: ", test_local_mae)
        
        return test_local_rmse, test_local_mae

    def evaluate_one_step(self):
        stats_test = self.test() 
        test_local_rmse = np.mean(stats_test[0])
        test_local_mae = np.mean(stats_test[1])
        self.rs_test_local_rmse.append(test_local_rmse)

        print(" ")
        print("Evaluate_Local_Model")
        print("Average Test local  RMSE: ", test_local_rmse)
        print("Average Test local  MAE: ", test_local_mae)

        return test_local_rmse, test_local_mae  
