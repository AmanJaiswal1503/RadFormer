# encoding: utf-8
"""
Training implementation
"""
import os
import cv2
import argparse
import numpy as np
import torch
import json
torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import GbDataSet, GbUsgDataSet
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from models import RadFormer
from PIL import Image

import wandb
wandb.require("core")
from tqdm import tqdm

# Set random seeds for reproducibility
import random
random.seed(42)
cv2.setRNGSeed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

N_CLASSES = 3
CLASS_NAMES = ['nrml', 'benign', 'malg']


def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--img_dir', dest="img_dir", default="data/gb_imgs")
    parser.add_argument('--train_list', dest="train_list", nargs='+', default="data/cls_split/train.txt")
    parser.add_argument('--val_list', dest="val_list", nargs='+', default="data/cls_split/val.txt")
    parser.add_argument('--out_channels', dest="out_channels", default=2048, type=int)
    parser.add_argument('--epochs', dest="epochs", default=30, type=int)
    parser.add_argument('--save_dir', dest="save_dir", default="experiments/replication")
    parser.add_argument('--save_name', dest="save_name", default="attnbag")
    parser.add_argument('--batch_size', dest="batch_size", default=16, type=int)
    parser.add_argument('--lr', dest="lr", default=0.001, type=float)
    parser.add_argument('--load_local', action="store_true")
    parser.add_argument('--pretrain', action="store_true")
    parser.add_argument('--global_net', dest="global_net", default="resnet50")
    parser.add_argument('--local_net', dest="local_net", default="bagnet33")
    parser.add_argument('--global_weight', dest="global_weight", default=0.6, type=float)
    parser.add_argument('--local_weight', dest="local_weight", default=0.1, type=float)
    parser.add_argument('--fusion_weight', dest="fusion_weight", default=0.3, type=float)
    parser.add_argument('--fusion_type', dest="fusion_type", default="+") #or *
    parser.add_argument('--optim', dest="optim", default="adam")
    parser.add_argument('--num_layers', dest="num_layers", default=2, type=int)
    parser.add_argument('--wandb_project', dest="wandb_project", default="radformer", type=str)
    parser.add_argument('--wandb_name', dest="wandb_name", default="replication", type=str)
    args = parser.parse_args()
    return args


def main(args):
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=args
    )

    print('********************load data********************')
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = GbUsgDataSet(data_dir=args.img_dir,
                            image_list_files=args.train_list,
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=0)
    
    val_dataset = GbUsgDataSet(data_dir=args.img_dir, 
                            image_list_files=args.val_list,
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, 
                                shuffle=False, num_workers=0)

    print('********************load data succeed!********************')


    print('********************load model********************')
    # initialize model
    model = RadFormer(local_net=args.local_net, \
                        global_weight=args.global_weight, \
                        local_weight=args.local_weight, \
                        fusion_weight=args.fusion_weight, \
                        load_local=args.load_local, use_rgb=True, \
                        num_layers=args.num_layers, pretrain=args.pretrain).cuda()

    #cudnn.benchmark = False
   
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    
    lr_sched = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 1)
    print('********************load model succeed!********************')

    print('********************initialisation results!********************')
    y_true, pred_g, pred_l, pred_f = validate(model, val_loader)
                
    acc_g, conf_g = log_stats(y_true, pred_g, label="Global")
    acc_l, conf_l = log_stats(y_true, pred_l, label="Local")
    acc_f, conf_f = log_stats(y_true, pred_f, label="Fusion")

    print('Initial Global Accuracy:', acc_g)
    print(conf_g)
    print('Initial Local Accuracy:', acc_l)
    print(conf_l)
    print('Initial Fusion Accuracy:', acc_f)
    print(conf_f)

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    save_model_name = args.save_name
    best_accs = [0, 0, 0]
    best_ep = 0

    print('********************begin training!********************')
    wandb.watch(model)

    global_step = 0
    validation_frequency = int(np.round(len(train_loader)/4))
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch , args.epochs - 1))
        print('-' * 10)
        #set the mode of model
        model.train()  #set model to training mode
        running_loss = 0.0
        #Iterate over data
        for i, (global_inp, target, filenames) in enumerate(tqdm(train_loader)):
            global_input_var = torch.autograd.Variable(global_inp.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            
            optimizer.zero_grad()
            loss = model(global_input_var, target_var) 
            loss.backward() 
            optimizer.step()  
            running_loss += loss.data.item()
            wandb.log({"train_loss": loss.item(), "epoch": epoch}, step=global_step)

            if (i+1) % validation_frequency == 0:
                val_loss = validate(model, val_loader, get_loss=True)
                wandb.log({"val_loss": val_loss, "epoch": epoch}, step=global_step)
                model.train()

            global_step += 1

        print('Loss: {:.5f}'.format(running_loss/len(train_loader)))

        print('*******validation*********')
        y_true, pred_g, pred_l, pred_f = validate(model, val_loader)
        
        #run["Train/Loss"].log(running_loss/len(train_loader))
        
        acc_g, conf_g = log_stats(y_true, pred_g, label="Global")
        acc_l, conf_l = log_stats(y_true, pred_l, label="Local")
        acc_f, conf_f = log_stats(y_true, pred_f, label="Fusion")
        wandb.log({"global_accuracy": acc_g, "local_accuracy": acc_l, "fusion_accuracy": acc_f, "epoch": epoch}, step=global_step)


        #save
        # torch.save(model.state_dict(), save_path+"/"+save_model_name+'_epoch_'+str(epoch)+'.pkl')
        if best_accs[0] < acc_f: #max(acc_g, acc_l, acc_f):
            best_accs = [acc_f, acc_l, acc_g] #max(acc_g, acc_l, acc_f)
            best_ep = epoch
            best_cfms = [conf_f, conf_l, conf_g]
            torch.save(model.state_dict(), save_path+"/"+save_model_name+'_best.pth')
            #print('Best acc model saved!')

        # LR schedular step
        lr_sched.step()  #about lr and gamma
    print("Best Epoch: ", best_ep)
    print("Fusion\n", best_cfms[0], best_accs[0])
    print("Global\n", best_cfms[2], best_accs[2])
    print("Local\n", best_cfms[1], best_accs[1])


def log_stats(y_true, y_pred, label="Eval"):
        acc = accuracy_score(y_true, y_pred)
        cfm = confusion_matrix(y_true, y_pred)
        
        #logobj["%s/Accuracy"%label].log(acc)
        #logobj["%s/Specificity"%label].log((cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1])))
        #logobj["%s/Sensitivity"%label].log(cfm[2][2]/np.sum(cfm[2]))
        
        return acc, cfm


def get_pred_label(pred_tensor):
    _, pred = torch.max(pred_tensor, dim=1)
    return pred.item()


def validate(model, val_loader, get_loss=False):
    model.eval()
    y_true, pred_g, pred_l, pred_f = [], [], [], []
    val_loss = 0
    for i, (global_inp, target, filenames) in enumerate(val_loader):
        with torch.no_grad():
            global_input_var = torch.autograd.Variable(global_inp.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            
            if get_loss:
                loss = model(global_input_var, target_var)
                val_loss += loss
            else:
                g_out, l_out, f_out, _ = model(global_input_var)
            
                y_true.append(target.tolist()[0])
                pred_g.append(get_pred_label(g_out))
                pred_l.append(get_pred_label(l_out)) 
                pred_f.append(get_pred_label(f_out)) 

    if get_loss:
        return val_loss/len(val_loader)
    else:
        return y_true, pred_g, pred_l, pred_f


if __name__ == "__main__":
    args = parse()
    main(args)
