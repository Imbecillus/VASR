print('TCD-TIMIT VISUAL SPEECH RECOGNIZER - EVALUATION\nBooting up...', flush=True)
print('Using ResNet10-LSTM', flush=True)

import os
import sys
import time
import torch
from torchvision import transforms
import viseme_list
import architectures
import TCDTIMITdataset as tcd
import helpers
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Standard values for model parameters
data_transforms = [
        transforms.Grayscale(),
        transforms.Resize((36, 36))
    ]
dataset_path = 'lips_phonemes_trainset.json'
validationset_path = None
n_files = None
truth_table = viseme_list.phonemes_38
savepath = './simple_cnn.pth'
viseme_set = None
channels = 1
device = torch.device("cpu")
batch_size = 1
choose_model = 'simple_CNN'
context = 0
weighted_loss = False
save_every = 2
eval_every = 1
dropout_rate = 0.0

if torch.cuda.is_available():
	print("CUDA available", flush=True)
	device = torch.device("cuda")
else:
    print("CUDA not available", flush=True)

# Parse command line arguments
for arg in sys.argv:
    if 'import=' in arg:
        savepath = arg[7:]
    if 'visemes=' in arg:
        viseme_set = arg[8:]
        if viseme_set == 'jeffersbarley':
            truth_table = viseme_list.visemes_jeffersbarley
        elif viseme_set == 'lee' in arg:
            truth_table = viseme_list.visemes_lee
        else:
            truth_table = viseme_list.visemes_neti
    if 'n_files=' in arg:
        n_files = int(arg[8:])
    if 'dataset=' in arg:
        dataset_path = arg[8:]
    if 'color=' in arg:
        if 'true' in arg:
            channels = 3
            data_transforms = [
                transforms.Resize((36,36))
            ]
        else:
            channels = 1
            data_transforms = [
                    transforms.Resize((36,36)),
                    transforms.Grayscale()
                ]
    if 'context=' in arg:
        context = int(arg[8:])
    if 'weighted_loss=true' in arg:
        weighted_loss = True
print('')

print('Loading model...', flush=True, end=' ')
from architectures import lstm
embedding_layer = lstm.ResNet(channels * (2 * context + 1), 128, (8, 16, 24, 32), dropout_rate, device).to(device)
model = lstm.Net(128, 128, len(truth_table), 1, embedding_layer).to(device)
model.load_state_dict(torch.load(savepath, map_location=device))
print('done.', flush=True)

print('Loading dataset...', flush=True, end=' ')
if weighted_loss:
    dataset = tcd.TCDTIMITDataset(dataset_path, n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context, sequences=True, truth='index')
else:
    dataset = tcd.TCDTIMITDataset(dataset_path, n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context, sequences=True)
print('done.', flush=True)

print('Evaluating over full validation set...', flush=True)

acc, classes, confusion_dict = helpers.lstm_evaluate(model, dataset, truth_table, 'index', device, print_confusion_matrix=True)
helpers.print_confusion_matrix(confusion_dict, truth_table, savepath)

print(f'Accuracy: {round(acc,2)}%')
print(f'{round(classes,2)}/{len(truth_table)} recognized.')
print('Confusion matrix has been exported.')
print('Evaluation finished.')

print('All done. :)')