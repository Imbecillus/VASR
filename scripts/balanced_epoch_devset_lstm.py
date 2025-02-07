# As balanced epoch devset, but with sequences per speaker instead of frames.

import json
import sys
import math
import random

dataset = sys.argv[1]
count = int(sys.argv[2])
verbose = True if '-v' in sys.argv else False

json_data = json.load(open(dataset, 'r'))

n_speakers = len(json_data.keys())
print(f'Speakers: {n_speakers}')
print(f'Saving {count} sequences per speaker.')

export_dict = {}

for speaker in json_data.keys():
    if len(json_data[speaker]) is not 0:
        if verbose:
            print(speaker)
        export_dict[speaker] = {}

        # Select count random frames from random sequences
        for i in range(count):
            sequences = list(json_data[speaker].keys())
            seq = random.choice(sequences)
            
            if seq not in export_dict[speaker]:
                export_dict[speaker][seq] = json_data[speaker][seq]

# Export
export_path = f"{dataset[0:-5]}_lstmbalanced.json"
print(f'Saving to {export_path}')
json.dump(export_dict, open(export_path, 'w'))
print('Done. :)')