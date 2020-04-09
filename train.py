# Training a lipreading CNN. Supports the following command line arguments:
# 'visemes=': Specify which viseme set shall be used. If no set is specified, the script trains to phonemes.
# 'n_files=': Specify how many files from the set shall be used.
# 'dataset=': path to the train set
# 'epochs=': how many epochs shall be trained
# 'color=': switch between 3-channel color images and 1-channel grayscale. Defaults to grayscale
# 'import=': specify a path of an exported model which the training shall continue from
# 'export=': specify the path of the exported model.
# '-i[X]': save intermediate steps every [X] epochs.
# 'context=': specify the amount of context to be given along with each frame. default is 0
# 'model=': specify model architecture to be trained
# 'weight_decay=': set weight decay

print('TCD-TIMIT VISUAL SPEECH RECOGNIZER - TRAINING\nBooting up...', flush=True)

import sys
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import TCDTIMITdataset as tcd
import helpers
import numpy as np
import matplotlib.pyplot as plt
import viseme_list

# Standard values for model parameters
data_transforms = [
        transforms.Grayscale(),
        transforms.Resize((36, 36))
    ]
dataset_path = 'lips_phonemes_trainset.json'
validationset_path = None
n_files = None
epochs = -1
truth_table = viseme_list.phonemes_38
savepath = './simple_cnn.pth'
cont_train = False
viseme_set = None
channels = 1
device = torch.device("cpu")
batch_size = 8
save_intermediate_models = False
perform_epoch_evaluation = False
choose_model = 'simple_CNN'
context = 0
wd = 0.0
weighted_loss = False
save_every = 2
eval_every = 1
dropout_rate = 0.0
learning_rate = 0.001
lr_warmup = False
lr_pt = None
threshold = None
epoch_evaluation_batch_size = 2048

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
        importpath = arg[7:]
        cont_train = True
        print('Continuing from saved model at', importpath)
    if 'import_offset=' in arg:
        offset = int(arg[14:])
    if 'export=' in arg:
        savepath = arg[7:]
    if '-i' in arg:
        save_intermediate_models = True
        if len(arg) > 2:
            save_every = int(arg[2:])
    if '-e' in arg:
        perform_epoch_evaluation = True
        if len(arg) > 2:
            eval_every = int(arg[2:])
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
    if 'devset=' in arg:
        validationset_path = arg[7:]
    if 'epoch_eval_count=' in arg:
        epoch_evaluation_batch_size = int(arg[17:])
    if 'epochs=' in arg:
        epochs = int(arg[7:])
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
    if 'learning_rate=' in arg:
        learning_rate = float(arg[14:])
    if 'lr_start=' in arg:
        lr_pt = float(arg[9:])
        lr_warmup = True
    if 'weight_decay=' in arg:
        wd = float(arg[13:])
    if 'weighted_loss=true' in arg:
        weighted_loss = True
    if 'dropout=' in arg:
        dropout_rate = float(arg[8:])
    if 'batch_size=' in arg:
        batch_size = int(arg[11:])
    if 'threshold=' in arg:
        threshold = float(arg[10:])
print('')

# Create folder with the same name as the export file for TensorBoard
writer = SummaryWriter(os.path.join('runs', os.path.basename(savepath[:-4])))

model = None

# Import chosen net architecture as 'architecture'
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
    model = model.to(device)
elif choose_model == 'DrResNet18':
    model = torchvision.models.resnet18()
    model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(512, len(truth_table)))
    model = model.to(device)
elif choose_model == 'ResNet10':
    from architectures import sigmedia as architecture
    model = architecture.Net(channels * (2 * context + 1), len(truth_table), 128, (8, 16, 24, 32), dropout_rate, device).to(device)

if not model:
    # channels = color channels of frame + color channels of the context frames
    model = architecture.Net(channels * (2 * context + 1), len(truth_table), dropout_rate).to(device)

if cont_train:
    # Load saved weights into the net
    model.load_state_dict(torch.load(importpath, map_location=device))

if weighted_loss:
    dataset = tcd.TCDTIMITDataset(dataset_path, n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context, truth='index')
    if validationset_path is not None:
        validationset = tcd.TCDTIMITDataset(validationset_path, n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context, truth='index')
else:
    dataset = tcd.TCDTIMITDataset(dataset_path, n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context)
    if validationset_path is not None:
        validationset = tcd.TCDTIMITDataset(validationset_path, n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context)

print('Starting new training (' + str(epochs) + ' epochs).')
if threshold == None:
    print('Aborting at convergence.')
else:
    print(f'Aborting at convergence or error below {threshold}.')
print('Model architecture: ' + choose_model)
print('Dropout: ', dropout_rate)
if weighted_loss:
    print('Loss Function: Cross Entropy Loss')
else:
    print('Loss Function: Mean Square Error')
print('Learning rate:' + str(learning_rate) + ';', "Warmup:", lr_warmup)
print('Train set: ' + dataset_path)
if not n_files is None:
    print('Choosing ' + str(n_files) + ' random files.')
else:
    print('Training over the full trainset (' + str(len(dataset)) + ').')
if validationset_path is not None:
    print(f'Evaluating over {epoch_evaluation_batch_size} validationset frames every {eval_every} epochs. (Set is {validationset_path}.)')
if channels == 1:
    print('Converting to grayscale.')
if truth_table == viseme_list.phonemes:
    print('Training for phonemes.')
else:
    print('Training for visemes (' + viseme_set + ').')
print(f'Exporting to {savepath} every {save_every} epochs.\n', flush=True)

# define training sequence
def loss_batch(model, loss_func, prediction, yb, opt=None):
    loss = loss_func(prediction, yb)        

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(prediction)

def fit(epochs, model, opt, train_dl, dataset, validationset):
    last_error = 100
    start = time.time()
    convergence_tracker = 0

    csv_export = "Epoch;Loss;TrainAcc;ValidAcc;Time\n"

    if weighted_loss:
        loss_func = torch.nn.CrossEntropyLoss(weight=weights)
        ground_truth = 'index'
    else:
        loss_func = torch.nn.MSELoss()
        ground_truth = 'one-hot'

    if lr_warmup:
        delta_lr = (learning_rate - lr_pt) / 4

    step = 1
    epoch = 0
    if cont_train:
        epoch = epoch + offset
    abort = False

    # Get batches for epoch evaluation
    epoch_eval_train = helpers.load_batch(dataset, epoch_evaluation_batch_size)
    epoch_eval_dev   = helpers.load_batch(validationset, epoch_evaluation_batch_size)

    while not abort:
        epoch_time = time.time()
        model.train()
        
        if lr_warmup and not epoch >= 5:
            # Learning Rate Warmup active, increasing lr
            lr = lr_pt + epoch * delta_lr
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        train_losses = []
        #stepstart = time.time()
        batchstart = time.time()
        
        for xb, yb in train_dl:
            #writer.add_scalar('batch load time', time.time() - stepstart, step)

            #stepstart = time.time()
            xb = xb.type(torch.float)
            tl = loss_batch(model, loss_func, model(xb.to(device)).to(device), yb.to(device), opt)
            
            #writer.add_scalar('batch loss calculation time', time.time() - stepstart, step)
            writer.add_scalar('batch time', time.time() - batchstart, step)
            writer.add_scalar('running loss', tl[0], step)
            train_losses.append(tl[0])

            step += 1
            #stepstart = time.time()
            batchstart = time.time()
        ll = 0
        ln = 0
        for l in train_losses:
            ll += l
            ln += 1
        training_loss = ll / ln
        writer.add_scalar('epoch loss', training_loss, epoch + 1)

        # Evaluate on train and dev set
        if perform_epoch_evaluation and ((epoch+1) % eval_every == 0 or epoch == 0):
            model.eval()
            train_acc, train_classes = helpers.batch_evaluate(epoch_eval_train, model, truth_table, ground_truth=ground_truth, device=device)
            train_acc = round(train_acc, 2)
            writer.add_scalar('train acc', train_acc, epoch + 1)

            valid_acc = ''
            valid_acc, valid_classes =  helpers.batch_evaluate(epoch_eval_dev, model, truth_table, ground_truth=ground_truth, device=device)
            valid_acc = round(valid_acc, 2)
            writer.add_scalar('valid acc', valid_acc, epoch + 1)

            # Print training loss, accuracies, and time for every epoch
            print(f"{epoch+1} === Err.: {round(training_loss, 4)}; Training Acc.: {train_acc} ({train_classes}/{len(truth_table)}); Valid. Acc.: {valid_acc} ({valid_classes}/{len(truth_table)}) === (Time: {helpers.time_since(start)} total, {helpers.time_since(epoch_time)} this epoch)", flush=True)
            csv_export = csv_export + f"{epoch+1};{round(training_loss, 4)};{train_acc};{valid_acc};{helpers.time_since(epoch_time)}\n".replace('.',',')
        else:
            # Print training loss and time for every epoch
            print(f"{epoch+1} === Err.: {round(training_loss, 4)} === (Time: {helpers.time_since(start)} total, {helpers.time_since(epoch_time)} this epoch)", flush=True)
            csv_export = csv_export + f"{epoch+1};{round(training_loss, 4)};;;{helpers.time_since(epoch_time)}\n".replace('.',',')

        epoch = epoch + 1

        # Abort if max epochs has been reached
        if epoch == epochs:
            print('Max epochs have been reached. Stopping training.')
            abort = True
        
        # Abort if loss hasn't changed for more than 3 epochs
        if abs(last_error - training_loss) < 0.00000001:
            convergence_tracker = convergence_tracker + 1
            if convergence_tracker > 1:
                print('Convergence. Stopping training.', flush=True)
                abort = True
        else:
            convergence_tracker = 0

        # Abort if threshold has been reached
        if threshold != None:
            if training_loss < threshold:
                print(f'Loss has sunken below threshold ({threshold}). Stopping training.')
                abort = True

        last_error = training_loss
        
        writer.add_scalar('epoch time', (time.time() - epoch_time) / 60, epoch)
        if save_intermediate_models and epoch % save_every == 0:
            torch.save(model.state_dict(), savepath[0:-4] + '_' + str(epoch) + '.pth')

    csv_f = open(savepath[0:-4] + '.csv', 'w')
    csv_f.write(csv_export)
    csv_f.close()

    print('\nValidating over the entire set...', flush=True)
    acc, confusion_matrix = helpers.evaluate(validationset, model, truth_table, device=device, verbose=True)
    print('Frame accuracy: ' + str(acc) + '\n')

train_batch_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

# Calculate weights
if weighted_loss:
    print('Calculating weights...', flush=True)
    import show_distribution
    distribution = show_distribution.count_distribution(dataset_path)
    top_class, _ = show_distribution.find_top_class(distribution)
    weight_dict = show_distribution.proportion_weight(distribution, top_class)
    for c in truth_table:
        if c not in weight_dict.keys():
            weight_dict[c] = 0
    weights = torch.tensor([weight_dict[c] for c in truth_table]).to(device)
    print(weight_dict, flush=True)

fit(epochs, model, opt, train_batch_dl, dataset, validationset)

# Save trained model
torch.save(model.state_dict(), savepath)

print('Finished Training\n', flush=True)

print('\nValidating over the entire set...', flush=True)
acc, confusion_matrix = helpers.evaluate(validationset, model, truth_table, device=device, verbose=True)
print('Frame accuracy: ' + str(acc) + '\n')