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

second_pass = False     # If any phoneme is missing in the conversion table, we need to perform a second pass

for speaker_key in json_data:
    for sequence_key in json_data[speaker_key]:
        for frame in json_data[speaker_key][sequence_key]:
            pair = json_data[speaker_key][sequence_key][frame]
            new_entry = conversion_table.get(pair[1], 'missing')
            json_data[speaker_key][sequence_key][frame][1] = new_entry
            if new_entry == 'missing':
                second_pass = True

if second_pass:
    def get_following_entry(data, frame_no):
        if data[frame_no][1] == 'missing':
            frame_no = int(frame_no) + 1
            frame_no = str(frame_no)
            return get_following_entry(data, frame_no)
        else:
            return data[frame_no][1]

    # On second pass, missing entries are replaced by the following viseme (since missing means, that the phoneme (e.g. /hh/) has no visual equivalent, meaning that the mouth is already forming the next viseme)
    print('  At least on entry was missing, perform second pass.')
    for speaker_key in json_data:
        for sequence_key in json_data[speaker_key]:
            for frame in json_data[speaker_key][sequence_key]:
                if json_data[speaker_key][sequence_key][frame][1] == 'missing':
                    json_data[speaker_key][sequence_key][frame][1] = get_following_entry(json_data[speaker_key][sequence_key], frame)
               
json.dump(json_data, open(export_path, 'w'))