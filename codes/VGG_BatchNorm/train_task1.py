import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np

from models.vgg import VGG_A,VGG_A_Dropout,VGG_A_Light,VGG_A_BatchNorm,get_number_of_parameters
from data.loaders import get_cifar_loader

def train(model,train_loader,val_loader,criterion,optimizer,device,scheduler=None,epochs=20):
    model=model.to(device)
    train_losses=[]
    val_accuracies=[]
    
    for epoch in range(epochs):
            model.train()
            running_loss=0.0
            for inputs,labels in train_loader:
                inputs,labels=inputs.to(device),labels.to(device)
                optimizer.zero_grad()
                outputs=model(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
            if scheduler:
                scheduler.step()
            train_losses.append(running_loss/len(train_loader))

            model.eval()
            correct=0
            total=0
            with torch.no_grad():
                for inputs,labels in val_loader:
                    inputs,labels=inputs.to(device),labels.to(device)
                    outputs=model(inputs)
                    _,predicted=torch.max(outputs.data,1)
                    total+=labels.size(0)
                    correct+=(predicted==labels).sum().item()
            val_acc=100*correct/total
            val_accuracies.append(val_acc)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_losses[-1]:.4f} | Val acc: {val_acc:.2f}%")
    return train_losses,val_accuracies
def plot_curve(values,title,ylabel,save_path):
        plt.figure()
        plt.plot(values,label=ylabel)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    
def main():
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader=get_cifar_loader(train=True,num_workers=0)
        val_loader=get_cifar_loader(train=False,num_workers=0)
        print("data loading finished,start training")

        os.makedirs("outputs",exist_ok=True)

        model_configs=[
            ("VGG_A",VGG_A),
            ("VGG_A_Dropout",VGG_A_Dropout),
            ("VGG_A_Light",VGG_A_Light),
            ("VGG_A_BatchNorm",VGG_A_BatchNorm)
        ]
        activations=[
            ("ReLU",nn.ReLU),
            ("LeakyReLU",nn.LeakyReLU)
        ]    
        optimizer_configs=[
            ("Adam",lambda params:optim.Adam(params,lr=0.001,weight_decay=1e-4)),
            ("SGD",lambda params:optim.SGD(params,lr=0.05,momentum=0.9,weight_decay=1e-4))
        ]
        criterion=nn.CrossEntropyLoss()
        for model_name,model_class in model_configs:
            for act_name,act_fn in activations:
                for opt_name,opt_fn in optimizer_configs:
                    tag=f"{model_name}_{act_name}_{opt_name}"
                    print(f"\n Training {tag}")

                    try:
                        model=model_class(activation=act_fn)
                    except TypeError:
                        model=model_class()
                    optimizer=opt_fn(model.parameters())
                    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
                    train_losses,val_accuracies=train(
                        model,train_loader,val_loader,criterion,optimizer,device,
                        scheduler=scheduler,epochs=20
                    )
                    params=get_number_of_parameters(model)
                    print(f"parameters of {tag}:{params:,}")
                    plot_curve(train_losses,f"{tag}-Training loss","Loss",f"outputs/{tag}_loss.png")
                    plot_curve(val_accuracies,f"{tag}-Validation Accuracy","Accuracy",f"outputs/{tag}_acc.png")
                    torch.save(model.state_dict(),f"outputs/{tag}_model.pth")
                    print(f"Saved:{tag}_model.pth & curve plots")
if __name__=="__main__":
       main()         
