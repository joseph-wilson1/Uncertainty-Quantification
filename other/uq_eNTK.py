### --- Dependencies --- ###
from definitions import *
import torch
import argparse
from tqdm import tqdm
import datetime

parser = argparse.ArgumentParser(description='PyTorch NTK Uncertainty Estimate Experiment')
parser.add_argument('--dataset',default='cifar',type=str,help='dataset (cifar or imagenet)')
args = parser.parse_args()

### --- CONSTANTS --- ###
dataset_str = 'energy'
TRAIN_RATIO = 0.9
BATCH_SIZE = 100
NORMALIZE_X = True
NORMALIZE_Y = True
LAYER_WIDTH = 50
NTK_WEIGHT_INIT = True

learning_rate = 1e-2
epochs = 400

ensemble_M = 5

FIND_KAPPA = True
REPORT_KAPPA = True
PYTORCH_KAPPA = False

### --- INPUT DATA HERE AS WELL AS DATASET NAME --- ###
if dataset_str == 'energy':
    df = pd.read_excel('.\data\Energy\ENB2012_data.xlsx')
    num_features = 8
elif dataset_str == 'concrete':
    df = pd.read_excel('.\data\Concrete\Concrete_Data.xls')
    num_features = 8

# print(data.shape)
print("--- Loading dataset {} --- \n".format(dataset_str))
print("Number of data points = {}".format(len(df)))
print("Number of coloumns = {}".format(len(df.columns)))
print("Number of features = {}".format(num_features))

training_set, test_set = data_split(df,TRAIN_RATIO)

if NORMALIZE_X:
    train_mX = training_set[:,:num_features].mean(axis=0)
    train_sX = training_set[:,:num_features].std(axis=0)
    train_sX[train_sX==0]=1
else:
    train_mX = 0
    train_sX = 1

if NORMALIZE_Y:
    train_my = training_set[:,num_features].mean(axis=0)
    train_sy = training_set[:,num_features].std(axis=0)
    if train_sy==0:
        train_sy=1
else:
    train_my = 0
    train_sy = 1

train_dataset = RegressionDataset(training_set, input_dim=num_features, mX=train_mX, sX=train_sX, my=train_my, sy=train_sy)
test_dataset = RegressionDataset(test_set, input_dim=num_features, mX=train_mX, sX=train_sX)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_set), shuffle=False)

np.set_printoptions(suppress=True, precision=3)
print("\n Sample training point \n X: {}, \n y: {} \n".format(train_dataset.__getitem__(0)[0], train_dataset.__getitem__(0)[1]))
print("Sample test point \n X: {}, \n y: {} ".format(test_dataset.__getitem__(0)[0], test_dataset.__getitem__(0)[1]))

## Create Model
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

ntk_network = MLP(input_d=num_features,width=LAYER_WIDTH).to(device=device, dtype=torch.float64)
if NTK_WEIGHT_INIT:
    ntk_network.apply(weights_init)
print("Number of parameters in NTK network = {}".format(sum(p.numel() for p in ntk_network.parameters() if p.requires_grad)))

## Train NTK network
mse_loss = nn.MSELoss(reduction='mean')
ntk_optimizer = optimizer_shared(ntk_network, type='adam', learning_rate=learning_rate)

# Run the training loop
print("\n Training NTK network \n")
for epoch in tqdm(range(epochs)):
    ntk_train_mse  = training_loop_ntk(
        dataloader=train_loader,
        model=ntk_network, 
        optimizer=ntk_optimizer,
        loss_function=mse_loss,
        verbose=False)
    
    ntk_test_mse = test_loop_ntk(
        dataloader=test_loader,
        model = ntk_network,
        my=train_my,
        sy=train_sy,
        loss_function=mse_loss)
    
# Process is complete.
print('\n NTK Training process has finished.')
print("Final training MSE = {:.3f}".format(ntk_train_mse))
print("Final test MSE = {:.2f}".format(ntk_test_mse))

## 
ensemble_model_list = []
ensemble_opt_list = []
ensemble_mse_training_list = np.empty(ensemble_M)
ensemble_mse_test_list = np.empty(ensemble_M)
ensemble_nll_list = np.empty(ensemble_M)

NLL = CustomNLL()
for i in range(ensemble_M):
    ensemble_model_list.append(EnsembleNetwork(input_d=num_features, width=LAYER_WIDTH).to(device=device,dtype=torch.float64))
    ensemble_opt_list.append(optimizer_shared(ensemble_model_list[i], type='adam', learning_rate=learning_rate))

for i in range(ensemble_M):
    print(f"\n Training ensemble {i+1} ")
    for t in tqdm(range(epochs)):
        ensemble_train_mse = training_loop_ensemble(
            dataloader=train_loader, 
            model=ensemble_model_list[i], 
            optimizer=ensemble_opt_list[i], 
            loss_function=NLL,
            mse_loss= mse_loss,
            verbose=False)
        ensemble_test_mse, ensemble_test_nll = test_loop_ensemble(
            dataloader=test_loader, 
            model=ensemble_model_list[i], 
            my=train_my, 
            sy=train_sy, 
            mse_loss=mse_loss,
            loss_function=NLL)
    print("Done!")
    ensemble_mse_training_list[i] = ensemble_train_mse
    ensemble_mse_test_list[i] = ensemble_test_mse
    ensemble_nll_list[i] = ensemble_test_nll
print('\n Ensemble training process has finished.')

## Print training metrics
print("Final ntk training MSE = {:.3f}".format(ntk_train_mse))
print("Final ntk test MSE = {:.2f}".format(ntk_test_mse))
print("Final ensemble training MSE = {}".format(ensemble_mse_training_list))
print("Final ensemble test MSE = {}".format(ensemble_mse_test_list))

### --- NTK --- ###
if FIND_KAPPA:
    EPSILON = 0
    print("\n --- Finding Kappa --- \n")
    Kappa = ntk_matrix(train_dataset,train_dataset,model=ntk_network,optimizer=ntk_optimizer)
    Kappa = Kappa + EPSILON*np.eye(len(train_dataset))

if REPORT_KAPPA:
    print("\n--- Kappa (Manual) Summary --- \n")
    print("Regularising constant = {}".format(EPSILON))
    print("Condition number of Kappa = {:.2f}".format(np.linalg.cond(Kappa)))
    eigvals = np.linalg.eigvalsh(Kappa)
    print("Number of negative eigenvalues of Kappa = {}".format(eigvals[eigvals<0].size))
    print("Number of zero eigenvalues of Kappa = {}".format(eigvals[eigvals==0].size))
    print("Smallest eigenvalue is = {}".format(sorted(eigvals)[0]))

### - PyTorch NTK --- ###

fnet, params = make_functional(ntk_network)

def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)

def empirical_ntk(params, x1, x2):
    # Compute J(x1)
    jac1 = torch.vmap(torch.func.jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]
    
    # Compute J(x2)
    jac2 = torch.vmap(torch.func.jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

def empirical_ntk_full(dataset, params):
    ntk = np.empty((len(dataset),len(dataset)))
    with tqdm(total=int(len(dataset)*len(dataset))) as pbar:
        for idx, x1 in enumerate(dataset):
            if type(x1) is tuple:
                x1,_ = x1
            for idy, x2 in enumerate(dataset):
                if type(x2) is tuple:
                    x2,_ = x2
                x1 = torch.reshape(x1,(1,-1))
                x2 = torch.reshape(x2,(1,-1))
                ntk[idx,idy] = empirical_ntk(params, x1, x2)
                pbar.update(1)
    return ntk

if PYTORCH_KAPPA:
    pytorch_Kappa = empirical_ntk_full(train_dataset, params)

    print("\n--- Kappa (PyTorch) Summary --- \n")
    print("Condition number of Kappa = {:.2f}".format(np.linalg.cond(pytorch_Kappa)))
    pytorch_eigvals = np.linalg.eigvalsh(pytorch_Kappa)
    print("Number of negative eigenvalues of Kappa = {}".format(pytorch_eigvals[pytorch_eigvals<0].size))
    print("Number of zero eigenvalues of Kappa = {}".format(pytorch_eigvals[pytorch_eigvals==0].size))
    print("Smallest eigenvalue is = {}".format(sorted(pytorch_eigvals)[0]))
    print("Frobenius norm difference between two methods is {:.2f}".format(
        np.linalg.norm(Kappa-pytorch_Kappa)
    )) 

    plot_dir = "./ntk_conditioning_results/"
    results_name = "results.txt"
    with open(plot_dir+results_name,'w') as results:
        results.write("{} \n".format(datetime.now()))
        results.write("Manual method: \
                    Condition number = {}, \
                    Num negative eigvals = {}, \
                    Smallest eigval = {}. \n".format(np.linalg.cond(Kappa),
                                                        eigvals[eigvals<0].size,
                                                        sorted(eigvals)[0])
        )
        results.write("PyTorch method: \
                    Condition number = {}, \
                    Num negative eigvals = {}, \
                    Smallest eigval = {}. \n".format(np.linalg.cond(pytorch_Kappa),
                                                        pytorch_eigvals[pytorch_eigvals<0].size,
                                                        sorted(pytorch_eigvals)[0])
        )
        results.write("Frobenius norm difference between two methods = {}".format(
            np.linalg.norm(Kappa-pytorch_Kappa)
        ))

## Uncertainty Quantification
print("\n --- Finding uncertainty estimates --- \n")
uncertainty_array = ntk_uncertainty_explicit(
    Kappa=Kappa, 
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    model=ntk_network,
    optimizer=ntk_optimizer,
    type='direct')
print("Finished! \n")
print("Number of zero values uncertainty array: {}".format(uncertainty_array[uncertainty_array==0].size))
print("Number of negative values for full rank: {}".format(uncertainty_array[uncertainty_array<0].size))

mu_mean, sigma_mean = ensemble_result(
    test_loader=test_loader,
    ensemble_M=ensemble_M,
    model_list=ensemble_model_list,
    sy=train_sy,
    my=train_my
)

### --- Plot results --- ###
print("\n --- Plotting Results --- \n")
today = datetime.date.today()
plot_dir = "./data/{}/{}/plot/".format(dataset_str, today.strftime("%d_%m_%Y"))
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

# ## NTK Scatter
# plot_name = "NTK_scatter.pdf"
# plt.scatter(final_ntk_rel_error, uncertainty_array*(train_sy**2))
# plt.xlabel("Relative squared error")
# plt.ylabel("Uncertainty")
# plt.title("NTK Method")
# plt.savefig(plot_dir + plot_name, format="pdf", bbox_inches="tight")
# plt.show()

# ## Deep Ensemble scatter
# plot_name = "Deep_ensemble_scatter.pdf"
# plt.scatter(Ensemble_rel_error_mean, sigma_mean)
# plt.xlabel("Relative squared error")
# plt.ylabel("Uncertainty")
# plt.title("Deep Ensemble")
# plt.savefig(plot_dir + plot_name, format="pdf", bbox_inches="tight")
# plt.show()

## NTK Histogram
plot_name = "ntk_hist.pdf"
plt.hist(sorted((train_sy**2)*uncertainty_array.squeeze(0)), bins='auto')
plt.xlabel("$\sigma^2$")
plt.ylabel("Frequency")
plt.title("Histogram of NTK uncertainty estimates")
plt.savefig(plot_dir + plot_name, format="pdf", bbox_inches="tight")

## Deep Ensemble Histogram
plot_name = "ensemble_hist.pdf"
plt.hist(sorted(sigma_mean), bins='auto')
plt.xlabel("$\sigma^2$")
plt.ylabel("Frequency")
plt.title("Histogram of Deep Ensemble uncertainty estimates")
plt.savefig(plot_dir + plot_name.format(dataset_str), format="pdf", bbox_inches="tight")

### --- Plot calibration curve --- ###

plot_name = "calibration_curve.pdf"
observed_true_ntk = calibration_curve_ntk(
    testloader=test_loader, 
    uncertainties=uncertainty_array, 
    model=ntk_network, 
    num_c=11, 
    my=train_my, 
    sy=train_sy)
observed_true_ensemble = calibration_curve_ensemble(
    testloader=test_loader, 
    mu=mu_mean, 
    sigma2=sigma_mean, 
    num_c=11)
plot_calibration(
    observed_true_ntk=observed_true_ntk, 
    observed_true_ensemble=observed_true_ensemble, 
    dataset_str=dataset_str, 
    dir_name=plot_dir, 
    plot_name=plot_name)

result_dir = "./data/{}/{}/result/".format(dataset_str, today.strftime("%d_%m_%Y"))
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

print("Final ntk training MSE = {:.3f}".format(ntk_train_mse))
print("Final ntk test MSE = {:.2f}".format(ntk_test_mse))
print("Final ensemble training MSE = {}".format(ensemble_mse_training_list))
print("Final ensemble test MSE = {}".format(ensemble_mse_test_list))

results_name = "results.txt"
with open(plot_dir+results_name,'w') as results:
    results.write("--- Results --- \n")
    results.write("Final ntk training mse = {:.3f}".format(ntk_train_mse))
    results.write("Final ntk test mse = {:.2f}".format(ntk_test_mse))
    results.write("Deep Ensemble training mse: mean = {:.2f}, std = {:.2f} \n".format(np.mean(ensemble_mse_training_list),np.std(ensemble_mse_training_list)))
    results.write("Deep Ensemble test mse: mean = {:.2f}, std = {:.2f} \n".format(np.mean(ensemble_mse_test_list),np.std(ensemble_mse_test_list)))
    results.write("Deep Ensemble NLL: mean = {:.2f}, std {:.2f} \n".format(np.mean(ensemble_nll_list),np.std(ensemble_nll_list)))
    results.write("\n --- Training Details --- \n")
    results.write("Learning Rate = {} \n".format(learning_rate))
    results.write("Training Epochs = {} \n".format(epochs))
    results.write("Number of ensembles = {} \n".format(ensemble_M))

    results.write("\n --- NTK Method Details --- \n")
    results.write("Regularising constant = {:.2f} \n".format(EPSILON))
    results.write("Condition number of Kappa = {:.2f} \n".format(np.linalg.cond(Kappa)))
    results.write("Number of negative eigenvalues of Kappa = {} \n".format(eigvals[eigvals<0].size))
    results.write("Number of zero eigenvalues of Kappa = {} \n".format(eigvals[eigvals==0].size))
    results.write("Number of negative values in NTK uncertainties array: {} \n".format(uncertainty_array[uncertainty_array<0].size))
    results.write("Number of zero values in NTK uncertainties array: {} \n".format(uncertainty_array[uncertainty_array==0].size))