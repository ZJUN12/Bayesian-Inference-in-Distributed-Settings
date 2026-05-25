import torch
import os
import h5py
import copy
from torch.nn import Module
import numpy as np

class Server:
    def __init__(self, K, dataset, algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, device):

        # Set up the main attributes
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        print("serverbase_type_model:",model)
        self.model = copy.deepcopy(model) if isinstance(model, Module) else model().to(device)  # global model.
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_global_rmse, self.rs_train_local_rmse, self.rs_test_global_rmse, self.rs_test_local_rmse, self.rs_train_global_loss, self.rs_train_local_loss = [], [], [], [], [], []
        self.times = times
 #-------------------the way of aggregating gradients---------------------       
    def aggregate_grads(self): # operation for aggregating gradient
        assert (self.users is not None and len(self.users) > 0)  
        # assert() is a checking function, this is used to judge if client is None
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
            # intialize the grad of each parameter, and its' shape is identical to param data
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)
    def add_grad(self, user, ratio): # the train_samples ratio as the aggregate weight
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio
            # the server model params is equal to 
    # broadcast global model to every client
    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
    
#-------------------the way of aggregating parameters--------------------- 
    # update the parameters of global model
    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio
    # aggregate parameters in server from every users 
    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # compute the aggregate weight of every users
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
    # save every global model to a path
    def save_model(self, post_fix_str):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_name = os.path.join(model_path, "server_" + post_fix_str + ".pt")
        torch.save(self.model, file_name)
        return file_name
    # loading global model from a path
    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)
    # check if exists of the "server.pt"
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
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        
        return self.users



    def persionalized_update_parameters(self,user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 file
    def save_results(self, post_fix_str):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        alg = alg + "_" + str(self.times)
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_test_local_rmse) != 0 & len(self.rs_train_local_rmse) & len(self.rs_train_local_loss)) :
            with h5py.File("/home/ovo/WZJ/FedBayes/results/"+'{}.h5'.format(alg + '_' + post_fix_str, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_per_rmse', data=self.rs_test_local_rmse)
                hf.create_dataset('rs_glob_rmse', data=self.rs_test_global_rmse)
                hf.create_dataset('rs_train_rmse', data=self.rs_train_local_rmse)
                hf.create_dataset('rs_train_loss', data=self.rs_train_local_loss)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_local_rmse = []
        tot_local_mae = []
        tot_local_epe = []
        Users_Predicted_mean_HI = []
        Users_Predicted_std_HI = []
        Users_True_HI = []
        for c in self.users:
            local_rmse, local_mae, local_epe, ns, Pre_mean_HI, Pre_std_HI, True_HI = c.test()
            tot_local_rmse.append(local_rmse)
            tot_local_mae.append(local_mae)
            tot_local_epe.append(local_epe)
            
            num_samples.append(ns)
            Users_Predicted_mean_HI.append(Pre_mean_HI)
            Users_Predicted_std_HI.append(Pre_std_HI)
            Users_True_HI.append(True_HI)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_local_rmse, tot_local_mae, tot_local_epe, 0, Users_Predicted_mean_HI, Users_Predicted_std_HI, Users_True_HI

    def testBayes(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_rmse = []
        tot_mae = []

        for c in self.users:
            c_rmse, c_mae, ns = c.testBayes()
            tot_rmse.append(c_rmse*1.0)
            tot_mae.append(c_mae*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_rmse, tot_mae

    def testpFedbayes(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_local_rmse = []
        tot_local_mae = []
        tot_local_epe = []
        tot_global_rmse = []
        tot_global_mae = []
        tot_global_epe = []
        Users_Predicted_mean_HI = []
        Users_Predicted_std_HI = []
        Users_True_HI = []
        for c in self.users:
            # return all_local_rmse, all_global_rmse, y.shape[0], predictions, uncertainties, True_HI
            local_rmse, local_mae, local_epe, global_rmse, glbal_mae, global_epe, ns, Pre_mean_HI, Pre_std_HI, True_HI = c.testpFedbayes()
            tot_local_rmse.append(local_rmse)
            tot_local_mae.append(local_mae)
            tot_local_epe.append(local_epe)
            tot_global_rmse.append(global_rmse)
            tot_global_mae.append(glbal_mae)
            tot_global_epe.append(global_epe)
            num_samples.append(ns)
            Users_Predicted_mean_HI.append(Pre_mean_HI)
            Users_Predicted_std_HI.append(Pre_std_HI)
            Users_True_HI.append(True_HI)
    
        return tot_local_rmse, tot_local_mae, tot_local_epe, tot_global_rmse, tot_global_mae, tot_global_epe, num_samples, Users_Predicted_mean_HI, Users_Predicted_std_HI, Users_True_HI


    def train_error_and_loss(self):
        num_samples = []
        tot_rmse = []
        tot_mae = []
        tot_epe = []
        losses = []
        for c in self.users:
            local_rmse, local_mae, local_epe, c_loss, ns = c.train_error_and_loss() 
            tot_rmse.append(local_rmse)
            tot_mae.append(local_mae)
            tot_epe.append(local_epe)
            num_samples.append(ns)
            losses.append(c_loss)
        
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_rmse, tot_mae, tot_epe, losses

    def train_error_and_loss_bayes(self):
        num_samples = []
        tot_rmse = []
        tot_mae = []
        losses = []
        for c in self.users:
            c_rmse, c_mae, c_loss, ns = c.train_error_and_loss_bayes()
            tot_rmse.append(c_rmse * 1.0)
            tot_mae.append(c_mae * 1.0)
            num_samples.append(ns)
            losses.append(c_loss * 1.0)

        ids = [c.id for c in self.users]

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
        for c in self.users:
            local_rmse, local_mae, local_epe, global_rmse, global_mae, global_epe, c_local_loss, c_global_loss, c_local_samples, c_global_samples = c.train_error_and_loss_pFedbayes()
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

        ids = [c.id for c in self.users]

        return tot_local_rmse, tot_local_mae, tot_local_epe,  tot_global_rmse, tot_global_mae, tot_global_epe, local_losses, global_losses, tot_local_samples, tot_global_samples
    

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_local_rmse = []
        tot_local_mae = []
        tot_local_epe = []
        Users_Predicted_mean_HI = []
        Users_Predicted_std_HI = []
        Users_True_HI = []
        for c in self.users:
            local_rmse, local_mae, local_epe, ns, Pre_mean_HI, Pre_std_HI, True_HI = c.test_persionalized_model()
            tot_local_rmse.append(local_rmse)
            tot_local_mae.append(local_mae)
            tot_local_epe.append(local_epe)
            num_samples.append(ns)
            Users_Predicted_mean_HI.append(Pre_mean_HI)
            Users_Predicted_std_HI.append(Pre_std_HI)
            Users_True_HI.append(True_HI)
        ids = [c.id for c in self.users]

        return num_samples, tot_local_rmse, tot_local_mae, tot_local_epe, Users_Predicted_mean_HI, Users_Predicted_std_HI, Users_True_HI
    
    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_rmse = []
        tot_mae = []
        tot_epe = []
        losses = []

        for c in self.users:
            local_rmse, local_mae, local_epe, c_loss, ns = c.train_error_and_loss_persionalized_model() 
            tot_rmse.append(local_rmse)
            tot_mae.append(local_mae)
            tot_epe.append(local_epe)
            num_samples.append(ns)
            losses.append(c_loss)
        
        ids = [c.id for c in self.users]

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

        print("Average Train Local RMSE: ", train_local_rmse)
        print("Average Train Local MAE: ", train_local_mae)
        
        print("Average Test Local RMSE: ", test_local_rmse)
        print("Average Test Local MAE: ", test_local_mae)
        Average_Test_Local_RMSE.append(test_local_rmse)
        Average_Test_Local_MAE.append(test_local_mae)
        
        return stats_test[6], stats_test[7], stats_test[8]
          
    def evaluate_pFedbayes(self):
        stats_test = self.testpFedbayes() 
        stats_train = self.train_error_and_loss_pFedbayes() 

        test_local_rmse = np.array([sum(x) for x in stats_test[0]]) / np.array(stats_test[6])
        test_local_mae = np.array([sum(x) for x in stats_test[1]]) / np.array(stats_test[6])
        test_local_epe = np.array([sum(x) for x in stats_test[2]]) / np.array(stats_test[6])
        train_local_rmse = np.array([sum(x) for x in stats_train[0]]) / np.array(stats_train[8])
        train_local_mae = np.array([sum(x) for x in stats_train[1]]) / np.array(stats_train[8])
        train_local_epe = np.array([sum(x) for x in stats_train[2]]) / np.array(stats_train[8])
        self.rs_test_local_rmse.append(test_local_rmse)
        self.rs_train_local_rmse.append(train_local_rmse)
        
        print("Evaluate_Local_Model")
        print(" ")
        print("Average Train local  RMSE: ", train_local_rmse)
        print("Average Train local  MAE: ", train_local_mae)
        print("Average Test local  RMSE: ", test_local_rmse)
        print("Average Test local  MAE: ", test_local_mae)

        return stats_test[7], stats_test[8], stats_test[9], test_local_rmse, test_local_mae  

    def evaluate_personalized_model(self):
        stats_test = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model() # return ids, num_samples, tot_rmse, tot_mae, losses
        test_local_rmse = np.array([sum(x) for x in stats_test[1]]) / stats_test[0]
        test_local_mae = np.array([sum(x) for x in stats_test[2]]) / stats_test[0]
        test_local_epe = np.array([sum(x) for x in stats_test[3]]) / stats_test[0]
        train_local_rmse = np.array([sum(x) for x in stats_train[2]]) / stats_train[1]
        train_local_mae = np.array([sum(x) for x in stats_train[3]]) / stats_train[1]
        train_local_epe = np.array([sum(x) for x in stats_train[4]]) / stats_train[1]
        train_local_loss = np.array([sum(x) for x in stats_train[4]]) / stats_train[1]
        self.rs_test_local_rmse.append(test_local_rmse)
        self.rs_train_local_rmse.append(train_local_rmse)
        self.rs_train_local_loss.append(train_local_loss)

        print("Average Train Local RMSE: ", train_local_rmse)
        print("Average Train Local MAE: ",train_local_mae)
        print("Average Train Local EPE: ",train_local_epe)
        
        print("Average Test Local RMSE: ", test_local_rmse)
        print("Average Test Local MAE: ", test_local_mae)
        print("Average Test Local EPE: ",test_local_epe)
        
        return stats_test[4], stats_test[5], stats_test[6],

    def evaluate_one_step(self):
        stats_test = self.test() 
        stats_train = self.train_error_and_loss() # return ids, num_samples, tot_rmse, tot_mae, losses
        test_local_rmse = np.array([sum(x) for x in stats_test[2]]) / stats_test[1]
        test_local_mae = np.array([sum(x) for x in stats_test[3]]) / stats_test[1]
        test_local_epe = np.array([sum(x) for x in stats_test[4]]) / stats_test[1]
        train_local_rmse = np.array([sum(x) for x in stats_train[2]]) / stats_train[1]
        train_local_mae = np.array([sum(x) for x in stats_train[3]]) / stats_train[1]
        train_local_epe = np.array([sum(x) for x in stats_train[4]]) / stats_train[1]
        self.rs_test_global_rmse.append(test_local_rmse)
        self.rs_train_local_rmse.append(train_local_rmse)

        print("Average Train Local RMSE: ", train_local_rmse)
        print("Average Train Local MAE: ", train_local_mae)
        print("Average Train Local EPE: ",train_local_epe)
        
        print("Average Test Local RMSE: ", test_local_rmse)
        print("Average Test Local MAE: ", test_local_mae)
        print("Average Test Local EPE: ",test_local_epe)
        
        return stats_test[6], stats_test[7], stats_test[8]
