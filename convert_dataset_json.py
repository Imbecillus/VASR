import json
import sys
import viseme_list

dataset_file = sys.argv[1]
if sys.argv[2] == 'Lee':
    conversion_table = viseme_list.phonemes_to_lee
elif sys.argv[2] == 'Neti':
    conversion_table = viseme_list.phonemes_to_neti
else:
    conversion_table = viseme_list.phonemes_to_jeffersbarley
export_path = sys.argv[3]

print('Converting', dataset_file, 'to', sys.argv[2], 'visemes.')
print('Saving to ', export_path)

json_data = json.load(open(dataset_file, 'r'))

for speaker_key in json_data:
    for sequence_key in json_data[speaker_key]:
        for frame in json_data[speaker_key][sequence_key]:
            pair = json_data[speaker_key][sequence_key][frame]
            pair[1] = conversion_table.get(pair[1], 'missing')

json.dump(json_data, open(export_path, 'w'))