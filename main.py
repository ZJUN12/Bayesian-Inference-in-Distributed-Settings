from algorithms.servers.serverpFedbayes import pFedBayes
from algorithms.trainmodel.models import *
from utils import arg
from utils.plot_utils import *
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, num_sites, K, personal_learning_rate, times, device,
         weight_scale, rho_offset, zeta, num_units, scenario, alpha):
    post_fix_str = 'plr_{}_lr_{}'.format(personal_learning_rate, learning_rate)
    model_path = []
    
    for i in range(times):
        torch.manual_seed(i)
        print("---------------Running time:------------", i)      
        if algorithm == "pFedBayes":
            # model = pBNN(1,120,2, device, weight_scale, rho_offset, zeta).to(device), model # scenario II
            model = pBNN(1,100,2, device, weight_scale, rho_offset, zeta).to(device), model # scenario I
            server = pFedBayes(K, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                           local_epochs, optimizer, num_sites, i, device, personal_learning_rate, 
                           num_units, scenario, alpha, post_fix_str=post_fix_str)
            
        model_path.append(server.train())
    
    result_path = average_data(num_users=num_sites, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta=beta, algorithms=algorithm, batch_size=batch_size,
                               dataset=dataset, k=K, personal_learning_rate=personal_learning_rate, times=times,
                               post_fix_str=post_fix_str)
    return model_path, result_path

def run():
    parser = arg.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="RUL", choices=["RUL"])
    parser.add_argument("--model", type=str, default="pbnn", choices=["pbnn"])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=0.001,
    help="Local learning rate")
    # scenario I
    parser.add_argument("--weight_scale", type=float, default=0.1) #pBNN:mod   
    parser.add_argument("--rho_offset", type=int, default=-5) #pBNN:model parameter scale
    # scenario II
    # parser.add_argument("--weight_scale", type=float, default=0.1) #pBNN:mod                                                                                                el weight initialization
    # parser.add_argument("--rho_offset", type=int, default=-7) #pBNN:model parameter scale
    parser.add_argument("--zeta", type=int, default=0.1) # pFedBayes/pFedSBayes: personalized coefficient
    parser.add_argument("--beta", type=float, default=1, # pFedDNN: client weight
    help="Average moving parameter for pFedMe")
    parser.add_argument("--lamda", type=int, default=2, help="Regularization term") #pFedMe: personalized coefficient
    parser.add_argument("--num_global_iters", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=20)                                            
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedBayes",
    choices=["pFedMe", "pFedBayes", "pFedDNN", "pFedSBayes"])
    ###############################################################################
    parser.add_argument("--num_sites", type=int, default=3, help="Number of Sites")
    parser.add_argument("--num_units", type=int, default=20, help="Number of Units of Each Site")
    parser.add_argument("--scenario", type=str, default="I", choices=["I:linear", "II:nonlinear"])
    parser.add_argument("--alpha", type=int, default=0.7, help="from different point start to predict") 
    ###############################################################################
    parser.add_argument("--K", type=int, default=3, help="Computation steps") #pFedMe: personalized training step
    parser.add_argument("--personal_learning_rate", type=float, default=0.001,
    help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Summary of training process:")
    print("Num_Sites: {}".format(args.num_sites))
    print("Num_Units: {}".format(args.num_units))
    print("Scenario: {}".format(args.scenario))
    print("Alpha: {}".format(args.alpha))
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_sites))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    return main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lamda=args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        num_sites=args.num_sites,
        num_units=args.num_units,
        scenario=args.scenario,
        alpha=args.alpha,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        device=device,
        weight_scale=args.weight_scale,
        rho_offset=args.rho_offset,
        zeta=args.zeta
    )

if __name__ == "__main__":
    run()