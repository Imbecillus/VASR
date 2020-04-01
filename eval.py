import sys
import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch import nn

import TCDTIMITdataset as tcd
import helpers
import viseme_list

print('TCD-TIMIT VISUAL SPEECH RECOGNIZER - EVALUATION', flush=True)
start = time.time()

# Standard values for model parameters
data_transforms = [
        transforms.Grayscale(),
        transforms.Resize((36, 36))
    ]
validationset_path = None
n_files = None
truth_table = viseme_list.phonemes_38
savepath = './net.pth'
viseme_set = None
channels = 1
device = torch.device("cpu")
choose_model = 'simple_CNN'
context = 0
suffix = None

if torch.cuda.is_available():
	print("CUDA available", flush=True)
	device = torch.device("cuda")
else:
    print("CUDA not available", flush=True)

# Parse command line arguments
for arg in sys.argv:
    if 'model=' in arg:
        choose_model = arg[6:]
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
    if 'validationset=' in arg:
        validationset_path = arg[14:]
    if 'color=' in arg:
        if 'true' in arg:
            channels = 3
            data_transforms = [
                transforms.Resize((100,100))
            ]
        else:
            channels = 1
            data_transforms = [
                    transforms.Grayscale(),
                    transforms.Resize((100, 100))
                ]
    if 'context=' in arg:
        context = int(arg[8:])
    if 'suffix=' in arg:
        suffix = arg[7:]

assert os.path.exists(validationset_path), 'Specified split file does not exist'

print('')

# Import chosen net architecture as 'architecture'
model = None
if choose_model == 'simple_CNN':
    from architectures import simple_CNN as architecture
elif choose_model == 'ConvNet':
    from architectures import lipreading_in_the_wild_convnet as architecture
elif choose_model == 'UnnConvNet':
    from architectures import lipreading_in_the_wild_convnet_unnormalized as architecture
elif choose_model == 'RNN-ConvNet':
    from architectures import rnn_convnet as architecture
elif choose_model == 'ResNet18':
    #data_transforms.append(transforms.Resize((256,256)))
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, len(truth_table))
elif choose_model == 'DrResNet18':
    data_transforms.append(transforms.Resize((256,256)))
    from architectures import ResNet18_dropout as architecture
elif choose_model == 'DrResNet18b':
    data_transforms.append(transforms.Resize((256,256)))
    model = torchvision.models.resnet18()
    model.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(512, len(truth_table)), nn.Softmax(dim=1))

# VALIDATION
testset = tcd.TCDTIMITDataset(validationset_path, data_transforms=data_transforms, n_files=n_files, viseme_set=viseme_set, context=context)

print('Starting evaluation for ' + savepath, flush=True)
print('Validation set: ' + validationset_path)

# Load saved net
if not model:
    model = architecture.Net(channels * (2 * context + 1), len(truth_table))
model.load_state_dict(torch.load(savepath, map_location=device))

# Print number of parameters trained
n = 0
total = 0
for p in model.parameters():
    nn = 1
    for s in list(p.size()):    # multiply dimension of tensor to get the total number of elements
        nn *= s
    total += nn
print('Number of parameters:', total)

# print example predictions and ground truths
print('Printing 8 random frame truths, predictions and output vectors...')
test_dl = DataLoader(testset, batch_size=8, shuffle=True)
dataiter = iter(test_dl)
image, label = dataiter.next()

#output = model(image)
#_, prediction = torch.max(output, 1)
#_, label = torch.max(label, 1)
#for t, p, o in zip(label, prediction, output):
#    print('Truth: ' + truth_table[t] + '; prediction: ' + truth_table[p])
#    print('Output: ', o)

print('\nValidating over the entire set...', flush=True)
acc, confusion_matrix = helpers.evaluate(testset, model, truth_table, device=device, verbose=True)
print('Frame accuracy: ' + str(acc) + '\n')

# If specified, add the suffix to the savepath
if suffix is not None:
    savepath = savepath[:-4] + '_' + suffix + '.png'
helpers.print_confusion_matrix(confusion_matrix, truth_table, savepath)

print('All done. Time:', helpers.time_since(start))
print('\n\n\n')