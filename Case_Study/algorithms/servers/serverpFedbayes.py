import torch
from tqdm import tqdm
from algorithms.users.userpFedbayes import UserpFedBayes
from algorithms.servers.serverbase import Server
from utils.data_utils import read_user_data
import numpy as np
import matplotlib.pyplot as plt



class pFedBayes(Server):
    def __init__(self,K, dataset,algorithm, model, batch_size, global_learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_clients, times, device, local_learning_rate,
                 output_dim=5, post_fix_str=''):
        super().__init__(K, dataset, algorithm, model[0], batch_size, global_learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_clients, times, device)

        self.local_learning_rate = local_learning_rate
        self.post_fix_str = post_fix_str
        total_clients = {"Client_Shanghai", "Client_Tianjin", "Client_Hubei"}
        print('clients initializting...')
        for client in total_clients:
            train, test = read_user_data(client)
            user = UserpFedBayes(K ,id, train, test, model, batch_size, global_learning_rate,beta,lamda, local_epochs, optimizer,
                                 local_learning_rate, device, output_dim=output_dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", num_clients, " / " ,total_clients)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        all_RMSE = np.empty((0, 3))
        all_MAE = np.empty((0, 3))
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.send_parameters()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs)
            self.aggregate_parameters()
            
            _, _, _, RMSE, MAE= self.evaluate_pFedbayes()

            print('ServerPFedB_RMSE_Shape:', RMSE.shape)
            print('ServerPFedB_MAE_Shape:', MAE.shape)
            print('ServerPFedB_RMSE_Type:', type(RMSE))
            print('ServerPFedB_MAE_Type:', type(MAE))
            print('ServerPFedB_RMSE_Length:', len(RMSE))
            print('ServerPFedB_MAE_Length:', len(MAE))

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
        plt.savefig('./FedBayes/images/RMSE_pFedBayes.pdf')
        plt.show()

        plt.figure(figsize=(10, 5))
        for i in range(all_MAE.shape[1]):
            plt.plot(time_steps, all_MAE[:, i], label=f'Wind Farm {i+1}')
        plt.title('MAE for Different Wind Farms')
        plt.xlabel('Iters')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig('./FedBayes/images/MAE_pFedBayes.pdf')
        plt.show()

        self.save_results(self.post_fix_str)
        return self.save_model(self.post_fix_str)
