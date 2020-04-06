# This script reads a confusion matrix and calculates two frame accuracies: one overall, one for only non-silence phonemes/visemes
# Usage: python calculate_nonsil_accuracies.py [path to csv] [jeffers/lee/phonemes]

import sys
csv_path = sys.argv[1]

if sys.argv[2] == 'phonemes':
    silence = 'sil'
    from viseme_list import phonemes_38 as classes
elif sys.argv[2] == 'jeffers':
    silence = 'S'
    from viseme_list import visemes_jeffersbarley as classes
elif sys.argv[2] == 'lee':
    silence = 'S'
    from viseme_list import visemes_lee as classes
else:
    raise NotImplementedError

confusion_matrix = []

with open(csv_path, 'r') as csv_file:
    line = csv_file.readline()

    while line:
        line = line[0:-2]   # Cut off newline
        if line == '':
            continue

        conf_matr_row = []

        entries = line.split(';')
        for number in entries:
            conf_matr_row.append(int(number))

        confusion_matrix.append(entries)
        line = csv_file.readline()

n_all = 0
n_nosil = 0
n_correct = 0
n_correct_no_sil = 0

for i in range(len(classes)):
    for j in range(len(classes)):
        cases = int(confusion_matrix[i][j])
        n_all = n_all + cases
        if i > 1:
            n_nosil = n_nosil + cases
        if i == j:
            n_correct = n_correct + cases
            if i > 1:
                n_correct_no_sil = n_correct_no_sil + cases

acc = round(n_correct / n_all * 100, 2)
acc_nosil = round(n_correct_no_sil / n_nosil * 100, 2)

print(f'Accuracy over all frames:  {acc}%')
print(f'Accuracy over non-silence: {acc_nosil}%')