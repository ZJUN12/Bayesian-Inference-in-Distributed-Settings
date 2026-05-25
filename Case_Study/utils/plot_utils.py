import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import os
import pandas as pd


def compare_pre_and_true(Users_Predicted_mean_HI, Users_Predicted_std_HI, Users_True_HI, plot_uncertainty=False):
    for i in range(len(Users_Predicted_mean_HI)):
        fig, ax = plt.subplots(figsize=(10, 6))
        predicted_mean = np.array(Users_Predicted_mean_HI[i])  
        predicted_std = np.array(Users_Predicted_std_HI[i])    
        true_data = np.array(Users_True_HI[i])
        
        predicted_mean = predicted_mean.squeeze(axis=0)  
        predicted_std = predicted_std.squeeze(axis=0)
        true_data = true_data.squeeze(axis=0)
    
        ax.plot(predicted_mean, label=f'Predicted Mean HI {i+1}', color='blue')

        ax.plot(true_data, label=f'True HI {i+1}', color='red', linestyle='--')
        
        ax.set_title(f'Wind Farm {i+1} Prediction vs True')
        ax.set_xlabel('Time')
        ax.set_ylabel('Healthy Indicator Value')
        
        ax.legend()

        plt.savefig(f'./FedBayes/images/multi-input-single-output/{i}.pdf')
        plt.show()

def compare_pre_and_true_uncertainty(Users_Predicted_mean_HI, Users_Predicted_std_HI, Users_True_HI, plot_uncertainty=False):
    for i in range(len(Users_Predicted_mean_HI)):
        fig, ax = plt.subplots(figsize=(10, 6))

        predicted_mean = np.array(Users_Predicted_mean_HI[i])  
        predicted_std = np.array(Users_Predicted_std_HI[i])    
        true_data = np.array(Users_True_HI[i])
        
        predicted_mean = predicted_mean.squeeze(axis=0)  
        predicted_std = predicted_std.squeeze(axis=0)
        true_data = true_data.squeeze(axis=0)
    
     
        predicted_all = []
        true_all = []
        std_all = []
        for j in range(predicted_mean.shape[0]):
            predicted = predicted_mean[j, -1]  
            true = true_data[j, -1]  
            std = predicted_std[j, -1] 
            
            predicted_all.append(predicted)
            true_all.append(true)
            std_all.append(std)

        predicted_all = np.array(predicted_all)
        true_all = np.array(true_all)
        std_all = np.array(std_all)
        save_to_csv(predicted_all, true_all, std_all)
        

        lower_bound = predicted_all - 2 * std_all
        upper_bound = predicted_all + 2 * std_all
        
        ax.plot(predicted_all, label=f'Predicted Mean HI {i+1}', color='blue')

        ax.fill_between(
            range(len(predicted_all)), 
            lower_bound, upper_bound, 
            color='blue', alpha=0.3, label='95% Confidence Interval'
        )
        
        ax.plot(true_all, label=f'True HI {i+1}', color='red', linestyle='--')
        
        ax.set_title(f'Wind Farm {i+1} Prediction vs True')
        ax.set_xlabel('Time')
        ax.set_ylabel('Healthy Indicator Value')
        
        ax.legend()

        # 保存图表
        plt.savefig(f'./FedBayes/images/multi-input-single-output/{i}.pdf')
        plt.show()
        
        
def save_to_csv(mean, std, true, file_name='predictions_and_truth.csv'):
    print("Main_len_main:",len(mean))
    save_dir = './FedBayes/results/multi-input-single-output'
    os.makedirs(save_dir, exist_ok=True)
    
    base_name, ext = os.path.splitext(file_name)
    counter = 1
    new_file_name = os.path.join(save_dir, file_name)

    while os.path.exists(new_file_name):
        new_file_name = os.path.join(save_dir, f"{base_name}_{counter}{ext}")
        counter += 1

    all_means = []
    all_stds = []
    all_trues = []

    for m, s, t in zip(mean, std, true):
        if isinstance(m, list):
            all_means.extend([x[0] for x in m])
            all_stds.extend([x[0] for x in s])
            all_trues.extend([x[0] for x in t])
        else:
            all_means.extend(m.flatten())
            all_stds.extend(s.flatten())
            all_trues.extend(t.flatten())
    
    data = {
        'mean': all_means,
        'std': all_stds,
        'true': all_trues
    }
    
    df = pd.DataFrame(data)
    df.to_csv(new_file_name, index=False)
    print(f"Data saved to {new_file_name}")


def simple_read_data(alg, parent_path='./FedBayes/results/multi-input-single-output'):
    """
    h5 file read.
    @param parent_path:
    @param alg:
    @return:
    """
    print(alg)
    hf = h5py.File('{}/'.format(parent_path) + '{}.h5'.format(alg), 'r')
    rs_glob_rmse = np.array(hf.get('rs_glob_rmse')[:])
    rs_per_rmse = np.array(hf.get('rs_per_rmse')[:]) if hf.get('rs_per_rmse') is not None else np.zeros(
        shape=rs_glob_rmse.shape)
    rs_train_rmse = np.array(hf.get('rs_train_rmse')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    if len(rs_per_rmse) == 0:
        rs_per_rmse = [np.nan] * len(rs_glob_rmse)
    return rs_train_rmse, rs_train_loss, rs_glob_rmse, rs_per_rmse

def get_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],beta=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = []):
    Numb_Algs = len(algorithms_list)
    train_rmse = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_rmse = np.zeros((Numb_Algs, Numb_Glob_Iters))
    per_rmse = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])
        string_learning_rate = string_learning_rate + "_" +str(beta[i]) + "_" +str(lamb[i])
        if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i]) + "_plr_"+ str(personal_learning_rate[i])  + "_lr_"+ str(learning_rate[i])

        data = np.array(simple_read_data(dataset +"_"+ algorithms_list[i] + "_avg"))[:, :Numb_Glob_Iters]
        data = np.array(simple_read_data(dataset + "_" + algorithms_list[i] + "_avg"))[:, :Numb_Glob_Iters]

        train_rmse[i, :] = np.mean(data, axis=0)
        train_loss[i, :] = np.mean(data, axis=0)
        glob_rmse[i, :] = np.mean(data, axis=0)
        per_rmse[i, :] = np.mean(data, axis=0)
        algs_lbl[i] = algs_lbl[i]
    return glob_rmse, per_rmse, train_rmse, train_loss


def get_all_training_data_value(num_users=3, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0, beta=0,
                                algorithms="", batch_size=0, dataset="", k=0, personal_learning_rate=0, times=5,
                                post_fix_str=''):
    train_rmse = np.zeros((times, Numb_Glob_Iters))
    train_loss = np.zeros((times, Numb_Glob_Iters))
    glob_rmse = np.zeros((times, Numb_Glob_Iters))
    rs_per_rmse = np.zeros((times, Numb_Glob_Iters))
    algorithms_list = [algorithms] * times
    for i in range(times):
        string_learning_rate = str(learning_rate)
        string_learning_rate = string_learning_rate + "_" + str(beta) + "_" + str(lamb)
        algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(
            batch_size) + "b"  "_" + str(loc_ep1) + "_" + str(i) + "_" + post_fix_str

        
    return glob_rmse, rs_per_rmse, train_rmse, train_loss


def average_data(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb="", learning_rate="", beta="", algorithms="",
                 batch_size=0, dataset="", k="", personal_learning_rate="", times=5, post_fix_str=''):
    glob_rmse, rs_per_rmse, train_rmse, train_loss = get_all_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                              learning_rate, beta, algorithms,
                                                                              batch_size, dataset, k,
                                                                              personal_learning_rate, times,
                                                                              post_fix_str)
    glob_rmse_data = np.average(glob_rmse, axis=0)
    rs_per_rmse_data = np.average(rs_per_rmse, axis=0)
    train_rmse_data = np.average(train_rmse, axis=0)
    train_loss_data = np.average(train_loss, axis=0)
    min_global_rmse = []
    for i in range(times):
        min_global_rmse.append(glob_rmse[i].max())

    print("std:", np.std(min_global_rmse))
    print("Mean:", np.mean(min_global_rmse))
    alg = dataset + "_" + algorithms
    alg = alg + "_" + str(learning_rate) + "_" + str(beta) + "_" + str(lamb) + "_" + str(num_users) + "u" + "_" + str(
        batch_size) + "b" + "_" + str(loc_ep1)
    if algorithms == "pFedMe" or algorithms == "pFedMe_p":
        alg = alg + "_" + str(k) + "_" + str(personal_learning_rate)
    alg = alg + "_" + post_fix_str + "_" + "avg"
    if len(glob_rmse) != 0 & len(train_rmse) & len(train_loss):
        with h5py.File("/home/ovo/WZJ/FedBayes/results/multi-input-single-output" + '{}.h5'.format(alg, loc_ep1), 'w') as hf:
            hf.create_dataset('rs_glob_rmse', data=glob_rmse_data)
            hf.create_dataset('rs_per_rmse', data=rs_per_rmse_data)
            hf.create_dataset('rs_train_rmse', data=train_rmse_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            return hf.filename


def get_label_name(name):
    if name.startswith("pFedMe"):
        if name.startswith("pFedMe_p"):
            return "pFedMe"+ " (PM)"
        else:
            return "pFedMe"+ " (GM)"
    if name.startswith("pFedbayes"):
        return "pFedbayes"
    if name.startswith("PerAvg"):
        return "Per-FedAvg"
    if name.startswith("FedAvg"):
        return "FedAvg"
    if name.startswith("APFL"):
        return "APFL"

def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len<3:
        return data
    for i in range(len(data)):
        x = data[i]
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        results.append(y[window_len-1:])
    return np.array(results)


def plot_summary_one_figure_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta,
                                          algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_rmse_, per_rmse_, train_rmse, train_loss_ = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters,
                                                               lamb,learning_rate, beta, algorithms_list, batch_size,
                                                               dataset, k, personal_learning_rate)
    for i in range(Numb_Algs):
        print("max rmse:", glob_rmse_[i].max())
    glob_rmse = average_smooth(glob_rmse_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    per_rmse = average_smooth(per_rmse_, window='flat')

    linestyles = ['-', '--', '-.', '-', '--', '-.']
    linestyles = ['-', '-', '-', '-', '-', '-', '-']
    markers = ["o", "v", "s", "*", "x", "P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1, figsize=(5, 5))
    plt.title("$\mu-$" + "strongly convex")
    plt.grid(True)
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_rmse[i, 1:], linestyle=linestyles[i], label=label+'(GM)', linewidth=1, color=colors[i], marker=markers[i],
                 markevery=0.2, markersize=5)

        plt.plot(per_rmse[i, 1:], linestyle=linestyles[i+2], label=label+'(PM)', linewidth=1, color=colors[i+2], marker=markers[i+2],
                 markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.10, 0.95])  
    plt.savefig(dataset.upper() + "Convex_test_Com.pdf", bbox_inches="tight")
    plt.close()
