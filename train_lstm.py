print('TCD-TIMIT VISUAL SPEECH RECOGNIZER - TRAINING\nBooting up...', flush=True)
print('Using ResNet10-LSTM', flush=True)

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import viseme_list
import architectures
import TCDTIMITdataset as tcd
import helpers
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

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
choose_model = ''
bidirectional = False
lstm_layers = 1
device = torch.device("cpu")
batch_size = 8
save_intermediate_models = False
perform_epoch_evaluation = False
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
offset = 0
epoch_evaluation_size_limit = None
quick_epoch_evaluation = False
pretrained_cnn = False

if torch.cuda.is_available():
	print("CUDA available", flush=True)
	device = torch.device("cuda")
else:
    print("CUDA not available", flush=True)

# Parse command line arguments
for arg in sys.argv:
    if 'import=' in arg:
        importpath = arg[7:]
        cont_train = True
        print('Continuing from saved model at', importpath, flush=True)
    if 'import_offset=' in arg:
        offset = int(arg[14:])
    if 'export=' in arg:
        savepath = arg[7:]
    if 'model=' in arg:
        choose_model = arg[6:]
    if 'lstm_layers=' in arg:
        lstm_layers = int(arg[12:])
    if '-b' in arg:
        bidirectional = True
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
    if '-q' in arg:
        quick_epoch_evaluation = True
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
    if 'pretrained_cnn=' in arg:
        pretrained_cnn = True
        pretrained_cnn_path = arg[15:]
print('')

# Create folder with the same name as the export file for TensorBoard
writer = SummaryWriter(os.path.join('runs', os.path.basename(savepath[:-4])))

model = None

print(f'Loading model {choose_model}...', flush=True, end=' ')
from architectures import lstm
if choose_model == '3-stage':
    embedding_layer = lstm.ResNet(channels * (2 * context + 1), 128, (8, 16, 24, 32), dropout_rate, device).to(device)
    model = lstm.Net(128, 128, len(truth_table), lstm_layers, embedding_layer, bidirectional=bidirectional).to(device)
elif choose_model == '2-stage':
    embedding_layer = lstm.ResNet(channels * (2 * context + 1), 128, (8, 16, 24, 32), dropout_rate, device).to(device)
    model = lstm.DirectNet(128, len(truth_table), lstm_layers, embedding_layer, bidirectional=bidirectional).to(device)
else:
    print('unknown model type specified.\nAborting.')
    exit()

print('done.', flush=True)

if pretrained_cnn:
    # Load saved weights into CNN part of the net
    missing, unexpected = model.embedding_layer.load_state_dict(torch.load(pretrained_cnn_path, map_location=device), strict=False)
    print('Missing keys', missing)
    print('unexpected keys', unexpected)

if cont_train:
    # Load saved weights into the net
    print('Loading saved weights into the net...', flush=True, end=' ')
    model.load_state_dict(torch.load(importpath, map_location=device))
    print('done.', flush=True)

print('Loading dataset...', flush=True, end=' ')
dataset = tcd.TCDTIMITDataset(dataset_path, n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context, sequences=True, truth='index')
validationset = tcd.TCDTIMITDataset(validationset_path, n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, sequences=True, context=context, truth='index')
if quick_epoch_evaluation:
    dataset_epochval = tcd.TCDTIMITDataset(dataset_path.replace('.json', '_lstmbalanced.json'), n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context, sequences=True, truth='index')
    validationset_epochval = tcd.TCDTIMITDataset(dataset_path.replace('.json', '_lstmbalanced.json'), n_files=n_files, data_transforms=data_transforms, viseme_set=viseme_set, context=context, sequences=True, truth='index')
else:
    dataset_epochval = dataset
    validationset_epochval = validationset
print('done.', flush=True)

# Calculate weights
if weighted_loss:
    print('Calculating weights...', flush=True, end=' ')
    import show_distribution
    distribution = show_distribution.count_distribution(dataset_path)
    top_class, _ = show_distribution.find_top_class(distribution)
    weight_dict = show_distribution.proportion_weight(distribution, top_class)
    for c in truth_table:
        if c not in weight_dict.keys():
            weight_dict[c] = 0
    weights = torch.tensor([weight_dict[c] for c in truth_table]).to(device)
    print(weight_dict, flush=True)
    print('done.')

print('Preparing training...', flush=True, end=' ')
train_dl = DataLoader(dataset, shuffle=True)
if weighted_loss:
    loss_function = nn.CrossEntropyLoss(weight=weights)
else:
    loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
print('done.', flush=True)

print('\nStarting training...')
step = 1
epoch = 1 + offset
last_error = 1000
abort = False
csv_export = "Epoch;Loss;TrainAcc;ValidAcc;Time\n"

while not abort:
    epoch_time = time.time()
    print(f'{epoch}: ', end='', flush=True)
    model.train()

    train_losses = []
    batchstart = time.time()

    for xb, yb in train_dl:
        prediction = model(xb.squeeze(dim=0).to(device))
        yb = yb.squeeze()
        loss = loss_function(prediction, yb.to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        writer.add_scalar('batch time', time.time() - batchstart, step)
        writer.add_scalar('running loss', loss, step)

        step += 1

        batchstart = time.time()

    ll = 0
    ln = 0
    for l in train_losses:
        ll += l
        ln += 1
    training_loss = ll / ln
    writer.add_scalar('epoch loss', training_loss, epoch)
    print(f'{round(training_loss, 4)} // ', end='', flush=True)

    if perform_epoch_evaluation and ((epoch) % eval_every == 0 or epoch == 1):
        model.eval()

        train_acc, train_classes = helpers.lstm_evaluate(model, dataset_epochval, truth_table, ground_truth='index', device=device, limit=epoch_evaluation_size_limit)
        train_acc = round(train_acc, 2)
        writer.add_scalar('train acc', train_acc, epoch + 1)

        print(f'Train. Acc.: {train_acc} ({train_classes}/{len(truth_table)}) // ', end='', flush=True)

        valid_acc = ''
        valid_acc, valid_classes =  helpers.lstm_evaluate(model, validationset_epochval, truth_table, ground_truth='index', device=device, limit=epoch_evaluation_size_limit)
        valid_acc = round(valid_acc, 2)
        writer.add_scalar('valid acc', valid_acc, epoch + 1)

        print(f'Valid. Acc.: {valid_acc} ({valid_classes}/{len(truth_table)}) // ', end='', flush=True)

        csv_export = csv_export + f"{epoch+1};{round(training_loss, 4)};{train_acc};{valid_acc};{helpers.time_since(epoch_time)}\n".replace('.',',')
    else:
        csv_export = csv_export + f"{epoch+1};{round(training_loss, 4)};;;{helpers.time_since(epoch_time)}\n".replace('.',',')

    # Abort if max epochs have been reached
    if epoch == epochs:
        print('Max epochs have been reached. Stopping training.')
        abort = True

    # Abort if loss hasn't changed for more than 3 epochs
    if abs(last_error - training_loss) < 0.0005:
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
    print(helpers.time_since(epoch_time), flush=True)

    if save_intermediate_models and epoch % save_every == 0:
        torch.save(model.state_dict(), savepath[0:-4] + '_' + str(epoch) + '.pth')
    
    csv_f = open(savepath[0:-4] + '.csv', 'w')
    csv_f.write(csv_export)
    csv_f.close()

    epoch = epoch + 1

print('Training finished.')

torch.save(model.state_dict(), savepath)
print(f'Model saved to {savepath}.\n\n')

print('Evaluating over full validation set...', flush=True)

acc, classes, confusion_dict = helpers.lstm_evaluate(model, validationset, truth_table, 'index', device, print_confusion_matrix=True)
helpers.print_confusion_matrix(confusion_dict, truth_table, savepath)

print(f'Accuracy: {round(acc,2)}%')
print(f'{round(classes,2)}/{len(truth_table)} recognized.')
print('Confusion matrix has been exported.')
print('Evaluation finished.')

print('All done. :)')