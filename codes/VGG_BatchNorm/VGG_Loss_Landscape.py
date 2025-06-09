import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
'''device_id = [0,1,2,3]
num_workers = 4
batch_size = 128'''
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
os.makedirs(figures_path,exist_ok=True)
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
'''device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))'''



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
'''train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    #
    #
    #
    #
    ## --------------------
    break'''



# This function is used to calculate the accuracy of model classification
def get_accuracy(model,data_loader):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for x,y in data_loader:
            x,y=x.to(device),y.to(device)
            outputs=model(x)
            _,predicted=torch.max(outputs,1)
            total+=y.size(0)
            correct+=(predicted==y).sum().item()
    return 100*correct/total




# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    '''learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)'''
    losses_list = []
    grads = []
    val_accuracy_curve =[]
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad_norm_list = []  # use this to record the loss gradient of each step
        #learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            #
            #
            #
            #
            ## --------------------


            loss.backward()
            last_layer_weight=None
            for name,param in model.named_parameters():
                if 'classifier' in name and 'weight' in name:
                    last_layer_weight=param
            if last_layer_weight is not None and last_layer_weight.grad is not None:
                grad_norm_list.append(last_layer_weight.grad.norm().item())
            

            optimizer.step()
            loss_list.append(loss.item())

        losses_list.append(loss_list)
        grads.append(grad_norm_list)
        val_acc=get_accuracy(model,val_loader)
        val_accuracy_curve.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs_n}, Val acc:{val_acc:.2f}%")


    return losses_list

def run_landscape_analysis(model_class,label,train_loader,val_loader,learning_rates=[1e-3,2e-3,1e-4,5e-4]):
    print(f"\nRunning loss landscape analysis for {label}")
    losses_all=[]
    for lr in learning_rates:
        print(f"Training {label} with lr={lr}")
        set_random_seeds(2020,device=device)
        model=model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        criterion = nn.CrossEntropyLoss()
        losses= train(model, optimizer, criterion, train_loader, val_loader, epochs_n=20)
        epoch_loss_mean=[np.mean(epoch_loss) for epoch_loss in losses]
        losses_all.append(epoch_loss_mean)
    losses_all=np.array(losses_all)
    min_curve=np.min(losses_all,axis=0)
    max_curve=np.max(losses_all,axis=0)
    return min_curve,max_curve

def plot_comparison(min_bn,max_bn,min_nobn,max_nobn):
    epochs=np.arange(len(min_bn))
    plt.figure(figsize=(10,5))
    plt.fill_between(epochs,min_bn,max_bn,alpha=0.3,label="VGG_A_BatchNorm Range",color='skyblue')
    plt.plot(min_bn,label="VGG_A_BatchNorm Min",color="blue")
    plt.plot(max_bn,label="VGG_A_BatchNorm Max",color="blue",linestyle='--')
    plt.fill_between(epochs,min_nobn,max_nobn,alpha=0.3,label="VGG_A Range",color='lightcoral')
    plt.plot(min_nobn,label="VGG_A Min",color="red")
    plt.plot(max_nobn,label="VGG_A Max",color="red",linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Landscape Comparison:VGG_A vs VGG_A_BatchNorm")
    plt.legend()
    plt.savefig(os.path.join(figures_path,"loss_landscape_comparison.png"))
    plt.close()
    print("Saved: loss_landscape_comparison.png")

if __name__=='__main__':
    train_loader=get_cifar_loader(train=True,num_workers=0)
    val_loader=get_cifar_loader(train=False,num_workers=0)
    min_bn,max_bn=run_landscape_analysis(VGG_A_BatchNorm,"VGG_A_BatchNorm",train_loader,val_loader)
    min_nobn,max_nobn=run_landscape_analysis(VGG_A,"VGG_A",train_loader,val_loader)
    plot_comparison(min_bn,max_bn,min_nobn,max_nobn)
