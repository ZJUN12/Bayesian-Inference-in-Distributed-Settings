import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


L_SIM = 600
T_MAX = 60.0
delta_t = T_MAX / L_SIM
t_grid = np.arange(L_SIM) * delta_t

lam, rho = 0.001, 1.05
mu_b = np.array([2.5, 0.01, 0.01])
Sigma_b = np.array([
    [0.2, -4e-4, 7e-5],
    [-4e-4, 3e-6, 1e-7],
    [7e-5, 1e-7, 3e-6]
])
sigma_eps = np.sqrt(2)

def generate_fk_m(t, b, scenario):
    base = b[0] + b[1] * t**1.2 + b[2] * t**1.7
    if scenario == "I":
        z = 0.0
    elif scenario == "II":
        c = np.random.uniform(0.99, 1.01)
        d = np.random.uniform(0.18, 0.22)
        z = c * np.sin(d * t)
    else:
        raise ValueError("Unknown scenario")
    return base + z


def generate_yk_m(f_t, sigma_eps): 
    eps = np.random.normal(0, sigma_eps, size=len(f_t))
    return f_t + eps


def simulate_unit_curve(scenario, sigma_eps):
    b = np.random.multivariate_normal(mu_b, Sigma_b)
    f_t = generate_fk_m(t_grid, b, scenario)
    y_t = generate_yk_m(f_t, sigma_eps)
    return t_grid, y_t


def plot_site_units(site_id, raw_curves, scenario):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    plt.figure(figsize=(6, 6))
    for curve in raw_curves:
        plt.plot(t_grid, curve, alpha=0.6, linewidth=1)

    plt.xlabel(r"Month $t$")
    plt.ylabel(r"Degradation Signal $y(t)$")
    save_path = f"./fig/site_{site_id}_scenario_{scenario}.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show() 
    plt.close()


def process_sequence_data(data_array, device, isX):
    sequence_length = 30
    pred_length = 1
    output = []

    data_array = data_array.reshape(-1, 1)

    for i in range(len(data_array) - sequence_length - pred_length + 1):
        if isX:
            seq = data_array[i:i+sequence_length]
        else:
            seq = data_array[i+sequence_length:i+sequence_length+pred_length]

        tensor = torch.tensor(seq, dtype=torch.float32).to(device)
        output.append(tensor)

    return output


def generate_site_dataset(client_id, n_units, scenario, device, split_mode="temporal"):
    print(f"Generating Site {client_id}: {n_units} units | Scenario {scenario} | Mode: {split_mode}")

    raw_X_list = []
    raw_Y_list = []


    for _ in range(n_units):
        t_grid, y_obs = simulate_unit_curve(scenario, sigma_eps)
        raw_X_list.append(t_grid)
        raw_Y_list.append(y_obs)
    
    plot_site_units(client_id, raw_Y_list, scenario)
    all_X_values = np.concatenate(raw_X_list)
    all_Y_values = np.concatenate(raw_Y_list)

    max_X = np.max(all_X_values)  
    max_Y = np.max(all_Y_values)

    print("data_utils_max_X:", max_X)
    print("data_utils_max_Y:", max_Y)

    norm_X_list = [x / max_X for x in raw_X_list]
    norm_Y_list = [y / max_Y for y in raw_Y_list]

    total_units_train_data = [] 
    total_units_test_data = []

    for i in range(n_units):
        X = np.array(norm_X_list[i]).reshape(-1, 1)
        Y = np.array(norm_Y_list[i]).reshape(-1, 1)

        full_len = len(X)
        if split_mode == "temporal":
            split = int(0.7 * full_len) 
        elif split_mode == "train_only":
            split = full_len            
        elif split_mode == "test_only":
            split = 0                   
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

  
        X_tr, X_te = X[:split], X[split:]
        Y_tr, Y_te = Y[:split], Y[split:]

  
        unit_train_samples = []
        unit_test_samples = []


        if len(X_tr) > 0:
            tensor_X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
            tensor_Y_tr = torch.tensor(Y_tr, dtype=torch.float32).to(device)
            unit_train_samples = list(zip(tensor_X_tr, tensor_Y_tr))
        if len(X_te) > 0:
            tensor_X_te = torch.tensor(X_te, dtype=torch.float32).to(device)
            tensor_Y_te = torch.tensor(Y_te, dtype=torch.float32).to(device)
            unit_test_samples = list(zip(tensor_X_te, tensor_Y_te))

        total_units_train_data.append(unit_train_samples)
        total_units_test_data.append(unit_test_samples)

    print(f"Site {client_id} Final Structure:")
    if len(total_units_train_data) > 0:
        print(f"  - Unit 0 Train Len: {len(total_units_train_data[0])}")
        print(f"  - Unit 0 Test Len:  {len(total_units_test_data[0])}")

    return total_units_train_data, total_units_test_data, 


def read_user_data(client_index, n_units=None, scenario=None, device=None, split_mode="temporal", time=None):
    np.random.seed(time)
    print("data_util_scenario:", scenario)
    device = torch.device("cpu") if device is None else device
    client_id = int(client_index.split("_")[-1]) if isinstance(client_index, str) else int(client_index)

    return generate_site_dataset(
        client_id=client_id,
        n_units=n_units,
        scenario=scenario,
        device=device,
        split_mode=split_mode,
    )



if __name__ == "__main__":
    device = torch.device("cpu")

    target_site_id = 0 
    
    clients_config = [
        {"name": "Client_0", "units": 5, "scenario": "I"}, # 假设 Site 0 是非线性的
        {"name": "Client_1", "units": 5, "scenario": "I"},
        {"name": "Client_2", "units": 5, "scenario": "I"},
    ]

    print(">>> Start Generating the Dataset (Leave-One-Site-Out Protocol)...")

    for conf in clients_config:
        name = conf["name"]
        cid = int(name.split("_")[-1])
        
        if cid == target_site_id:
            mode = "test_only"  
        else:
            mode = "train_only" # 
            
        train, test = read_user_data(
            client_index=name, 
            n_units=conf["units"], 
            device=device, 
            scenario=conf["scenario"],
            split_mode=mode,  
            time=1,
        )
        
        print(f"[{name}] Mode: {mode:<10} | Train samples: {len(train):<5} | Test samples: {len(test)}")