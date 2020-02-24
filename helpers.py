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

def evaluate(validationset, model, truth_table, ground_truth='one-hot', device=None):
    import torch
    from torch.utils.data import DataLoader

    # Load everything onto the same device
    if device == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    model.to(device)

    valid_dl = DataLoader(validationset, shuffle=True)

    count_all = 0
    count_correct = 0
    confusion_matrix = {}
    certainty = {}

    step = int(0.1 * len(valid_dl))

    for xb, yb in valid_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        count_all += 1
        prediction_vector = model(xb).to(device)

        # Print progress counter
        if count_all % step == 0:
            if count_all != len(valid_dl):
                print(round(count_all / len(valid_dl), 2) * 100, '% :', round(100 * (count_correct / count_all), 2), '% Acc', flush=True)
            else:
                print(round(count_all / len(valid_dl), 2) * 100)

        # Get index of the likeliest prediction
        _, pred = torch.max(prediction_vector, 1)

        # Get index of the ground truth (if ground truth isn't already given as an index)
        if ground_truth != 'index':
            _, truth = torch.max(yb, 1)
        else:
            truth = yb

        if pred == truth:                                       # If prediction is correct
            count_correct += 1                                  # Count correct predictions up
            truth_phoneme = truth_table[truth]
            if truth_phoneme not in certainty.keys():           # Add phoneme/viseme to list of classes that were recognized
                certainty[truth_phoneme] = []
            score = prediction_vector[0][truth]
            certainty[truth_phoneme].append(float(score[0]))    # Save score, so that we can calculate the average prediction scores later

        # Confusion Matrix
        truth = truth_table[truth]
        pred = truth_table[pred]
        if truth not in confusion_matrix.keys():
            confusion_matrix[truth] = {}                        # Initialize entries in confusion matrix
        if pred not in confusion_matrix[truth].keys():
            confusion_matrix[truth][pred] = 0
        confusion_matrix[truth][pred] += 1                      # Increment entry in confusion matrix for the truth-prediction-pair

    # Print average prediction scores for recognized phonemes/visemes
    print('Average prediction scores (for phonemes/visemes that were recognized at least once):')
    for key in certainty.keys():
        avg_certainty = 0.0
        for c in certainty[key]:
            avg_certainty = avg_certainty + c
        avg_certainty = avg_certainty / len(certainty[key])
        print(key, avg_certainty)
    print('Total classes trained: ', len(certainty), '/', len(truth_table), flush=True)

    return 100 * (count_correct / count_all), confusion_matrix

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