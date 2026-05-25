import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def save_to_csv(mean, file_name='true_healthy_indicator.csv'):
    print("Main_len_main:",len(mean))
    save_dir = './FedBayes/results'
    os.makedirs(save_dir, exist_ok=True)
    
    base_name, ext = os.path.splitext(file_name)
    counter = 1
    new_file_name = os.path.join(save_dir, file_name)

    while os.path.exists(new_file_name):
        new_file_name = os.path.join(save_dir, f"{base_name}_{counter}{ext}")
        counter += 1

    data = {
        'true_all': mean,
    }
    
    df = pd.DataFrame(data)
    df.to_csv(new_file_name, index=False)
    print(f"Data saved to {new_file_name}")


def plot_healthy_indicator(data,index):
    plt.figure()
    plt.plot(data)
    plt.xlabel("Index")
    plt.ylabel("Healthy Indicator")
    plt.title(index)
    plt.savefig(f'./images/{index}.pdf')
    plt.show()
    
def degration(client,time):
    client['时间'] = pd.to_datetime(client['时间'], errors='coerce')  # 使用 errors='coerce' 处理无效日期
    if client['时间'].isnull().any():
        print("Warning")
    filter_time = pd.to_datetime(time)
    filtered_data = client[client['时间'] < filter_time]

    return filtered_data
    
def read_data(client_data, avg_temp, time, WINDOWS, index):
    data = degration(client_data, time)
    scaler = MinMaxScaler()
    data = data.copy()
    healthy_indicator = (data['低速轴承温度(℃)'] - data['低速轴承温度(℃)'].mean()).rolling(window=WINDOWS).mean()/((data['风轮转速(rpm)'])*0.3).rolling(window=WINDOWS).mean()
    healthy_indicator = healthy_indicator[10000:]
    healthy_indicator = scaler.fit_transform(healthy_indicator.values.reshape(-1, 1))
    experimental_data = data[:][['低速轴承温度(℃)', '风轮转速(rpm)']].rolling(window=WINDOWS).mean()
    experimental_data = experimental_data[10000:]
    experimental_data = scaler.fit_transform(experimental_data)

    
    plot_healthy_indicator(healthy_indicator,index)
    split_index = int(0.7 * len(experimental_data))
    X_train_data = experimental_data[:split_index]
    Y_train_data = healthy_indicator[:split_index]
    print('X_train_head:',X_train_data[:9])
    print('y_train_head:',Y_train_data[:9])


    print('data_utils_train_data_shape:',len(X_train_data))
    X_test_data  = experimental_data[split_index:]
    Y_test_data = healthy_indicator[:split_index]
    print('data_utils_test_data_shape:',len(X_test_data))
    
    return X_train_data, Y_train_data, X_test_data, Y_test_data


def read_user_data(index, device = None):

    device = torch.device('cuda') if device is None else device
    
    if index == "Client_Shanghai":
        Client_Shanghai = pd.read_csv('./data/WT35_shanghai/WT_35号风机.csv')
        X_train_data, Y_train_data, X_test_data, Y_test_data = read_data(Client_Shanghai, 25.8, '2019-08-01 11:01:12', 10000, index) # 2019-08-01 11:01:12
        X_train = process_sequence_data(X_train_data, device, True)
        y_train = process_sequence_data(Y_train_data, device, False)
        X_test = process_sequence_data(X_test_data, device, True)
        y_test = process_sequence_data(Y_test_data, device, False)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        
        return train_data, test_data
    
    elif index == "Client_Tianjin":
        Client_Tianjin = pd.read_csv('./data/WT1_tianjin/WT_1号.csv')
        X_train_data, Y_train_data, X_test_data, Y_test_data = read_data(Client_Tianjin, 9.6, '2019-05-24 13:34:20', 10000, index)# 2019-05-24 13:34:20
        X_train = process_sequence_data(X_train_data, device, True)
        y_train = process_sequence_data(Y_train_data, device, False)
        X_test = process_sequence_data(X_test_data, device, True)
        y_test = process_sequence_data(Y_test_data, device, False)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        
        return train_data, test_data
    
    else:
        Client_Hubei = pd.read_csv('./data/WT24_hubei/WT_ 3-333024.csv')
        X_train_data, Y_train_data, X_test_data, Y_test_data = read_data(Client_Hubei, 9.6, '2019-06-10 22:07:01', 10000, index) #2019-06-10 22:07:01
        X_train = process_sequence_data(X_train_data, device, True)
        y_train = process_sequence_data(Y_train_data, device, False)
        X_test = process_sequence_data(X_test_data, device, True)
        y_test = process_sequence_data(Y_test_data, device, False)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        
        return train_data, test_data
    

def process_sequence_data(data, device, isX):
    sequence_length = 30  
    pred_length = 1     
    X = []
    y = []
    data = np.array(data)  

    if isX == True:
        for i in range(len(data) - sequence_length - pred_length + 1):
            input_seq = data[i:i+sequence_length]
            X_tensor = torch.Tensor(input_seq).type(torch.float32).to(device)
            X.append(X_tensor)
        return X
    else:
        for i in range(len(data) - sequence_length - pred_length + 1):
            target_seq = data[i+sequence_length:i+sequence_length+pred_length]
            y_tensor = torch.Tensor(target_seq).type(torch.float32).to(device)
            y.append(y_tensor)
        return y


