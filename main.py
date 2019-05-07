import numpy as np
import pandas as pd 
import os
import cv2
import gc
import matplotlib.pyplot as plt
import random
import argparse
import json
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
import tqdm
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import albumentations
from albumentations import torch as AT

from . import models
from . import optimizers
from .loss import getLoss
from .reader import read_data, read_test

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model_conv, train_loader, epoch):
    model_conv.train()
    avg_loss = 0.
    avg_corrects = 0
    avg_auc = 0
    running_auc = 0
    running_corrects = 0
    train_top1_sum = 0

    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        global_feat, output_train = model_conv(imgs_train)
        output_train = output_train[:,0]
        loss = getLoss(global_feat, output_train, labels_train.float())
        
        loss.backward()
        optimizer.step() 
        avg_loss += loss.item() / len(train_loader)
        preds = torch.where(torch.sigmoid(output_train) > 0.5, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())
        running_corrects += torch.sum(preds == labels_train.float()).item()
        acc = running_corrects / (3* (idx+1) * train_loader.batch_size)
    
        auc = 0
        try:
            a = labels_train.data.cpu().numpy()
            b = output_train.detach().cpu().numpy()
            auc = roc_auc_score(a, b) 
            avg_auc += auc/ len(train_loader)
        except ValueError:
            pass         
        if (idx+1) % 100 == 0:
            print('{}/{} \t loss={:.4f}, acc={:.4f}, auc={:.4f}'.format(idx+1, \
                    len(train_loader), avg_loss*len(train_loader)/(idx+1), acc, auc))
            
    avg_corrects = running_corrects / len(train_loader) / train_loader.batch_size /3   
    return avg_loss, avg_corrects, avg_auc

def test(model_conv, valid_loader):
    avg_val_loss = 0.
    model_conv.eval()
    running_corrects = 0
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(valid_loader):
            imgs_vaild, labels_vaild = imgs.cuda(), labels.cuda()
            global_feat, output_test = model_conv(imgs_vaild)
            output_test = output_test[:,0]
            avg_val_loss += getLoss(global_feat, output_test, labels_vaild.float()).item() / len(valid_loader)
            preds = torch.where(torch.sigmoid(output_test) > 0.5, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())
            running_corrects += torch.sum(preds == labels_vaild.float()).item()
        avg_val_corrects = running_corrects / (3 * len(valid_loader) * valid_loader.batch_size)
    return avg_val_loss, avg_val_corrects    

def main(): 
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'predict'])
    arg('run_root')
    arg('trained_weight')
    arg('--model', default='densenet169')
    arg('--optimizer', default='adamw')
    arg('--pretrained', type=int, default=1)
    arg('--warmingup', type=int, default=1)
    arg('--batch-size', type=int, default=32)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    arg('--lr', type=float, default=1e-4)
    arg('--patience', type=int, default=3)
    arg('--epochs', type=int, default=100)
    arg('--epoch-size', type=int)
    arg('--clean', action='store_true')
    arg('--tta', type=int, default=4)
    arg('--debug', action='store_true')
    args = parser.parse_args()

    run_root = Path(args.run_root)
    trained_weight = Path(args.trained_weight)
    seed_everything(seed=2333)

    model_conv = getattr(models, args.model)(num_classes=N_CLASSES, pretrained=args.pretrained)
    model_conv.cuda()    
    optim = getattr(optimizers, args.optimizer)(params=model_conv.parameters())
               
    if run_root.exists() and args.clean:
        shutil.rmtree(run_root)
            
    run_root.mkdir(exist_ok=True, parents=True)
    (run_root / 'params.json').write_text(json.dumps(vars(args), indent=4, sort_keys=True))
    batch_size = args.batch_size,
    num_workers = args.workers
             
    if trained_weight.exists():
        if(os.path.exists(trained_weight)):
            model_conv.load_state_dict(torch.load(trained_weight), strict=False)
    
    if args.mode == 'train':

        train_loader, valid_loader = read_data(run_root, batch_size, num_workers)
        
        if args.warmingup:
            model_conv.freeze_basemodel()
            n_epochs = 1
            p = 0
            valid_loss_min = float("inf")
            optimizer = optim(lr=1e-2)
            for i in range(n_epochs):
                start_time = time.time()
                avg_loss, avg_corrects, avg_auc = train(i)
                avg_val_loss, avg_val_corrects = test()
                elapsed_time = time.time() - start_time  
                print('Epoch {}/{} \t loss={:.4f} acc={:.4f} auc={:.4f} \t val_loss={:.4f} val_acc={:.4f} \t time={:.2f}s'.format(\
                    i + 1, n_epochs, avg_loss, avg_corrects, avg_auc, avg_val_loss, avg_val_corrects, elapsed_time))

        optimizer = optim(lr=args.lr)    
        model_conv.unfreeze_model()
        n_epochs = args.epochs
        patience = args.patience
        for i in range(n_epochs):
            start_time = time.time()
            avg_loss, avg_corrects, avg_auc = train(model_conv, train_loader, i)
            avg_val_loss, avg_val_corrects = test(model_conv, valid_loader)
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} acc={:.4f} auc={:.4f} \t val_loss={:.4f} val_acc={:.4f} \t time={:.2f}s'.format(\
                i + 1, n_epochs, avg_loss, avg_corrects, avg_auc, avg_val_loss, avg_val_corrects, elapsed_time))

            if avg_val_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(\
                valid_loss_min,avg_val_loss))
                torch.save(model_conv.state_dict(), 'model.pt')
                valid_loss_min = avg_val_loss
                p = 0

            if avg_val_loss > valid_loss_min:
                p += 1
                print(f'{p} epochs of increasing val loss')
                if p >= 1:
                    print('Decrease learning rate')
                    optimizer = optim(lr=args.lr/10)  
                if p > patience:
                    print('Stopping training')
                    stop = True
                    break   
    else:
        num_tta = args.tta
        test_loader = read_test(root_path, batch_size, num_workers)
        model_conv.eval().cuda()

        for tta in range(num_tta):
            preds = []
            for batch_i, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = model_conv(data).detach()

                pr = output[:,0].cpu().numpy()
                for i in pr:
                    preds.append(i)

            test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})
            test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
            sub = pd.read_csv(f'{run_root}/data/sample_submission.csv')
            sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')
            sub = sub[['id', 'preds']]
            sub.columns = ['id', 'label']
            sub.head()
            sub.to_csv('single_model_'+str(tta)+'.csv', index=False)

        del model_conv
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()        