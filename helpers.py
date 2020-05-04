def time_since(start):
    import time
    now = time.time()
    passed = now - start
    unit = 'seconds'

    if passed > 60:
        passed = passed / 60
        unit = 'minutes'

    if passed > 60:
        passed = passed / 60
        unit = 'hours'

    return str(round(passed, 2)) + ' ' + unit

def evaluate(validationset, model, truth_table, ground_truth='one-hot', device=None, max=None, verbose=False):
    import torch
    from torch.utils.data import DataLoader

    batch_size = 1

    # Load everything onto the same device
    if device == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    model.to(device)

    valid_dl = DataLoader(validationset, batch_size=batch_size, shuffle=True)

    count_all = 0
    count_correct = 0
    confusion_matrix = {}
    certainty = {}

    step = int(0.1 * (len(valid_dl) * batch_size))

    for xb, yb in valid_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        predictions = model(xb).to(device)

        for i in range(len(predictions)):
            count_all += 1

            prediction_vector = predictions[i]
            label = yb[i]

            # Print progress counter
            if verbose:
                if count_all % step == 0:
                    if count_all != len(valid_dl):
                        print(round(count_all / (batch_size * len(valid_dl)), 2) * 100, '% :', round(100 * (count_correct / count_all), 2), '% Acc', flush=True)
                    else:
                        print(round(count_all / (batch_size * len(valid_dl)), 2) * 100)

            # Get index of the likeliest prediction
            _, pred = torch.max(prediction_vector, 0)

            # Get index of the ground truth (if ground truth isn't already given as an index)
            if ground_truth != 'index':
                _, truth = torch.max(label, 0)
            else:
                truth = label

            if pred == truth:                                       # If prediction is correct
                count_correct += 1                                  # Count correct predictions up
                truth_phoneme = truth_table[truth]
                if truth_phoneme not in certainty.keys():           # Add phoneme/viseme to list of classes that were recognized
                    certainty[truth_phoneme] = []
                score = prediction_vector[truth]
                certainty[truth_phoneme].append(float(score))    # Save score, so that we can calculate the average prediction scores later

            # Confusion Matrix
            truth = truth_table[truth]
            pred = truth_table[pred]
            if truth not in confusion_matrix.keys():
                confusion_matrix[truth] = {}                        # Initialize entries in confusion matrix
            if pred not in confusion_matrix[truth].keys():
                confusion_matrix[truth][pred] = 0
            confusion_matrix[truth][pred] += 1                      # Increment entry in confusion matrix for the truth-prediction-pair

        # Abort if a maximum number of samples to be evaluated has been specified and reached
        if max is not None:
            if count_all >= max:
                break

    # Print average prediction scores for recognized phonemes/visemes
    if verbose:
        print('Average prediction scores (for phonemes/visemes that were recognized at least once):')
        for key in certainty.keys():
            avg_certainty = 0.0
            for c in certainty[key]:
                avg_certainty = avg_certainty + c
            avg_certainty = avg_certainty / len(certainty[key])
            print(key, avg_certainty)
        print('Total classes trained: ', len(certainty), '/', len(truth_table), flush=True)

    if verbose:
        return 100 * (count_correct / count_all), confusion_matrix
    else:
        return 100 * (count_correct / count_all), len(certainty)

def batch_evaluate(batch, model, truth_table, ground_truth='one-hot', device=None):
    import torch

    # Load everything onto the same device
    if device == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    model = model.to(device)

    count_all = 0
    count_correct = 0
    recognized_classes = []

    inputs = batch[0].to(device)
    predictions = model(inputs).to(device)

    labels = batch[1].to(device)

    for i in range(len(inputs)):
        yb = labels[i]
        prediction_vector = predictions[i]

        count_all += 1

        # Get index of the likeliest prediction
        _, pred = torch.max(prediction_vector, 0)

        # Get index of the ground truth (if ground truth isn't already given as an index)
        if ground_truth != 'index':
            _, truth = torch.max(yb, 1)
        else:
            truth = yb

        if pred == truth:                                       # If prediction is correct
            count_correct += 1                                  # Count correct predictions up
            truth_phoneme = truth_table[truth]
            if truth_phoneme not in recognized_classes:           # Add phoneme/viseme to list of classes that were recognized
                recognized_classes.append(truth_phoneme)
            
    return 100 * (count_correct / count_all), len(recognized_classes)

def evaluate_lstm_batch(batch, model, truth_table, ground_truth = 'index', device=None, dct_feats=False):
    import torch

    count_all = 0
    count_correct = 0
    recognized_classes = []
    confusion_dict = {}

    inputs = batch[0]
    labels = batch[1].squeeze()

    if dct_feats:
        inputs = inputs.transpose(0,1)

    predictions = model(inputs.to(device))
    if dct_feats:
        predictions = predictions[0].squeeze()

    for i in range(len(inputs)):
        count_all += 1

        _, pred = torch.max(predictions[i], 0)

        if ground_truth != 'index':
            _, truth = torch.max(labels[i], 1)
        else:
            truth = labels[i]

        truth_phoneme = truth_table[truth]
        pred_phoneme = truth_table[pred]

        if truth_phoneme not in confusion_dict.keys():
            confusion_dict[truth_phoneme] = {}
        if pred_phoneme not in confusion_dict[truth_phoneme].keys():
            confusion_dict[truth_phoneme][pred_phoneme] = 1
        else:
            confusion_dict[truth_phoneme][pred_phoneme] += 1

        if pred == truth:
            count_correct += 1

            if truth_phoneme not in recognized_classes:           # Add phoneme/viseme to list of classes that were recognized
                 recognized_classes.append(truth_phoneme)

    return count_correct, count_all, len(recognized_classes), confusion_dict

def lstm_evaluate(model, set, truth_table, ground_truth = 'index', device=None, limit=None, print_confusion_matrix = False, dct_feats = False):
    import torch
    from torch.utils.data import DataLoader

    # Load everything onto the same device
    if device == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    if print_confusion_matrix:
        confusion_matrix = {}
        for truth in truth_table:
            confusion_matrix[truth] = {}
            for pred in truth_table:
                confusion_matrix[truth][pred] = 0

    count_all = 0
    count_correct = 0
    recognized_classes = 0

    dl = DataLoader(set, batch_size=1)
    dl_iter = iter(dl)

    for batch in dl_iter:
        correct, length, classes, confusion_dict = evaluate_lstm_batch(batch, model, truth_table, ground_truth = 'index', device=device, dct_feats=dct_feats)

        if classes > recognized_classes:
            recognized_classes = classes
        count_all += length
        count_correct += correct

        if print_confusion_matrix:
            for truth in truth_table:
                for pred in truth_table:
                    confusion_matrix[truth][pred] += confusion_dict.get(truth, {}).get(pred, 0)

        if limit is not None:
            if count_all > limit:
                break

    if print_confusion_matrix:
        return 100 * (count_correct / count_all), recognized_classes, confusion_matrix
    else:
        return 100 * (count_correct / count_all), recognized_classes

def load_batch(dataset, size):
    from torch.utils.data import DataLoader
    
    dl = DataLoader(dataset, batch_size=size)
    dl_iter = iter(dl)

    return next(dl_iter)

def load_lstm_batch(dataset, size):
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset)
    dl_iter = iter(dl)

    batch = next(dl_iter)
    batch[0] = batch[0].squeeze(dim=0)
    batch[1] = batch[1].squeeze(dim=0)

    return batch

def print_confusion_matrix(confusion_dict, truth_table, savepath):
    confusion_matrix = [[0 for x in range(len(truth_table))] for y in range(len(truth_table))]      # Initialize NxN matrix filled with zeros with N=length of truth table

    for x in range(len(truth_table)):
        truth = truth_table[x]
        if truth in confusion_dict.keys():
            for y in range(len(truth_table)):
                pred = truth_table[y]
                confusion_matrix[x][y] = confusion_dict[truth].get(pred, 0)                         # Write confusion value into the matrix; write 0 if it didn't occur
    
    # Save confusion matrix as csv
    with open(savepath[0:-4] + '_conf.csv', 'w') as output:                                         # ... and in csv format
        for x in range(len(truth_table)):
            for y in range(len(truth_table)):
                output.write(str(confusion_matrix[x][y]) + ';')
            output.write('\n')
        output.close()

    # Now, create and display/export the matrix as a graphic
    # Normalize each row of the matrix for the most common element in it
    for x in range(len(truth_table)):
        row = confusion_matrix[x]
        max_predictions = max(row)
        for y in range(len(truth_table)):
            confusion_matrix[x][y] /= max_predictions + 0.1

    import matplotlib.pyplot as plt
    import time
    plt.pcolormesh(confusion_matrix)                                                                # Create PyPlot diagram for confusion matrix
    plt.title('Confusion matrix')
    plt.xlabel('predictions')
    plt.ylabel('truths')
    locs = [i + 0.5 for i in range(len(truth_table))]
    plt.xticks(locs, truth_table, horizontalalignment='center')
    plt.yticks(locs, truth_table, verticalalignment='center')
    
    plt.savefig(savepath[0:-4] + '_conf.png', dpi=300)                                              # Save confusion matrix as an image...
    plt.show()