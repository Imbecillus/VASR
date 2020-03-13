import viseme_list
truth_table = viseme_list.visemes_jeffersbarley

import sys
import_path = sys.argv[1]
if len(sys.argv) < 3:
    export_path = import_path[:-4] + '_new.csv'
else:
    export_path = sys.argv[2]

confusion_matrix = [[0 for x in range(len(truth_table))] for y in range(len(truth_table))]      # Initialize NxN matrix filled with zeros with N=length of truth table

with open(import_path, 'r') as input:
    line = input.readline()
    x = 0
    while line:
        values = line.split(';')
        for y in range(len(truth_table)):
            confusion_matrix[x][y] = int(values[y])
        x += 1
        line = input.readline()

# Normalize each row of the matrix for the most common element in it
for x in range(len(truth_table)):
    row = confusion_matrix[x]
    max_predictions = max(row)
    for y in range(len(truth_table)):
        confusion_matrix[x][y] /= max_predictions

import matplotlib.pyplot as plt
import time
plt.pcolormesh(confusion_matrix)                                                                # Create PyPlot diagram for confusion matrix
plt.title('Confusion matrix')
plt.xlabel('predictions')
plt.ylabel('truths')
locs = [i + 0.5 for i in range(len(truth_table))]
plt.xticks(locs, truth_table, horizontalalignment='center')
plt.yticks(locs, truth_table, verticalalignment='center')

plt.savefig(export_path[0:-4] + '_conf.png', dpi=300)                                              # Save confusion matrix as an image...
plt.show()

# Save confusion matrix as csv
with open(export_path[0:-4] + '_conf.csv', 'w') as output:                                         # ... and in csv format
    for x in range(len(truth_table)):
        for y in range(len(truth_table)):
            output.write(str(confusion_matrix[x][y]) + ';')
        output.write('\n')
    output.close()
