#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:24:45 2022

@author: Zephyr
"""
import os
import torch
from tqdm import tqdm
from torch import nn

from models.pvig_gaze import pvig_ti_224_gelu
from read_data import read_mimic

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score
import numpy as np

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    
    label_all = []
    outputs_all = []
    
    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            x,labels,gaze = sample
            
            label_all.append(labels.numpy())
            
            x,labels,gaze = x.to(device), labels.to(device), gaze.to(device)

            outputs = model(x, gaze)
            loss = loss_fn(outputs, labels)
            _,pred = torch.max(outputs,1)
            num_correct = (pred == labels).sum()
            
            loss = loss.item()
            acc = num_correct.item()/len(labels)
            count += len(labels)
            test_loss += loss*len(labels)
            test_acc += num_correct.item()
            pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
            
            outputs_all.append(outputs.detach().cpu().numpy())
    label_all = np.concatenate(label_all)
    outputs_all = np.concatenate(outputs_all)
    return test_loss/count, test_acc/count, label_all, outputs_all

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    save_dir = '../output/gazegnn_add3_adam_rotate'
    data_dir = '../mimic_part_jpg'
    batchsize = 32

    
    _, test_generator = read_mimic(batchsize,data_dir)
    
    model = pvig_ti_224_gelu()
    model.prediction[-1] = nn.Conv2d(model.prediction[-1].in_channels, 3, 1, bias=True)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'acc_model.pth'), map_location=torch.device("mps")))
    print(model)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    test_loss, test_acc, y_test, y_score = test_loop(test_generator, model, criterion, device)
    print("\nTest loss: {:f}, acc: {:f}".format(test_loss, test_acc))   
    y_pred = y_score.argmax(axis=-1)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print("\nPrecision: {:f}, recall: {:f}, f1: {:f}".format(precision, recall, f1)) 
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_names = ['CHF', 'Normal', 'Pneumonia']
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10,10))    
    colors = ['C0', 'C1', 'C2']
    for i, color in zip([1, 0, 2], colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=4,
                 label=class_names[i]+'(AUC=%.3f)' % roc_auc[i])
        
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Average(AUC={roc_auc['macro']:.3f})",
        color='C4',
        linestyle=":",
        linewidth=8,
    )
    plt.plot([0, 1], [0, 1], 'k--', lw=4)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate',fontsize=25)
    plt.ylabel('True Positive Rate',fontsize=25)
    #plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right", fontsize=23)
    plt.tick_params(labelsize=25)
    plt.rcParams.update({'font.size': 25})
    plt.grid()
    plt.savefig('auc.eps', format='eps', bbox_inches='tight')
    plt.show()