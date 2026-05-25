import torch
from tqdm import tqdm
from algorithms.users.userpFedbayes import UserpFedBayes
from algorithms.servers.serverbase import Server
from utils.data_utils import read_user_data
import numpy as np
import matplotlib.pyplot as plt


class pFedBayes(Server):
    def __init__(self,K, dataset,algorithm, model, batch_size, global_learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_sites, times, device, local_learning_rate, num_units, scenario, alpha,
                 output_dim=5, post_fix_str=''):
        super().__init__(K, dataset, algorithm, model[0], batch_size, global_learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_sites, times, device, local_learning_rate, num_units, scenario, alpha,)
        self.times = times
        self.local_learning_rate = local_learning_rate
        self.post_fix_str = post_fix_str
        total_sites = [f"site_{i}" for i in range(num_sites)]
        print('sites initializting...')
        print("scenario:", scenario)
        target_test_site_id = 0
        for i, client_name in enumerate(total_sites):
            client_id = i

            if client_id == target_test_site_id:
                current_mode = "test_only"
                print(f" -> Set {client_name} as TARGET SITE (Test Only)")
            else:
                current_mode = "train_only"
            

            train, test = read_user_data(
                client_index=client_name, 
                n_units=num_units,           
                scenario=scenario,   
                device=device,
                split_mode=current_mode,      
                time = self.times
            )
            
            user = UserpFedBayes(
                K, 
                client_id, 
                train, 
                test, 
                model, 
                batch_size, 
                global_learning_rate,
                beta,
                lamda, 
                local_epochs, 
                optimizer,
                local_learning_rate, 
                device, 
                output_dim=output_dim
            )
            if current_mode == "train_only":  
                self.train_users.append(user)
                self.total_train_samples += user.train_samples
            else:
                self.test_users.append(user)      
                self.total_test_samples += user.test_samples
                print(f"User {client_id} added to test_users list.") 
            
        print("Number of users / total users:", num_sites, " / " ,total_sites)
        print(f"DEBUG Check: len(train_users)={len(self.train_users)}, len(test_users)={len(self.test_users)}")
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.train_users is not None and len(self.train_users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.train_users:
            user.set_grads(grads)

    def train(self):
        num_test_nodes = len(self.test_users)
        print("serverpFedbayes_len_test_users:", len(self.test_users))
        all_RMSE = np.empty((0, num_test_nodes))
        all_MAE = np.empty((0, num_test_nodes))
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.send_parameters()
            for user in self.train_users:
                user.train(self.local_epochs)
            self.aggregate_parameters()

            RMSE, MAE= self.evaluate_pFedbayes(glob_iter)

            all_RMSE = np.vstack((all_RMSE, RMSE))
            all_MAE = np.vstack((all_MAE, MAE))

        print('all_RMSE_shape:', all_RMSE.shape)
        print('all_MAE_shape:', all_MAE.shape)

        time_steps = np.arange(all_RMSE.shape[0])

        plt.figure(figsize=(10, 5))
        for i in range(all_RMSE.shape[1]):
            plt.plot(time_steps, all_RMSE[:, i], label=f'Wind Farm {i+1}')
        plt.title('RMSE for Different Wind Farms')
        plt.xlabel('Iters')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        plt.savefig('./Simulation_FedBayes/images/RMSE_pFedBayes.pdf')
        plt.show()


        plt.figure(figsize=(10, 5))
        for i in range(all_MAE.shape[1]):
            plt.plot(time_steps, all_MAE[:, i], label=f'Wind Farm {i+1}')
        plt.title('MAE for Different Wind Farms')
        plt.xlabel('Iters')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig('./Simulation_FedBayes/images/MAE_pFedBayes.pdf')
        plt.show()

        self.save_results(self.post_fix_str)
        return self.save_model(self.post_fix_str)
