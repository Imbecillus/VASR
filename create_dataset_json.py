import os
import json
import sys
import random

# This script will parse through the selected subset directories in the specified dataset folder and output a .json file containing the paths to all frames
dataset_path = "data"
subsets_to_include = "volunteers"   # can be volunteers, lipspeakers or all
truth = "phonemes"                  # specify whether phonemes or visemes are to be used as the ground truth?
create_traintestval_split = True    # specify whether seperate files for train set, test set and validation set are to be created
export_name = "splits/full"
decimate = False
roi = False 
speaker_selection = []
speakers_to_exclude = []
phonemes_to_exclude = []
#testset_speakers = ['28M', '55F', '25M', '56M', '49F', '44F', '33F', '09F', '18M', '54M', '45F', '36F', '34M', '15F', '58F', '08F', '41M']   # Recommended Train-Test-Split
#validationset_speakers = ['03F', '06M', '37F', '52M', '40F', '39M' '32F', '42M', '48M', '53M']
testset_speakers = ['44F', '45F', '49F', '55F', '58F', '34M', '41M', '54M', '56M']
validationset_speakers = ['08F', '09F', '15F', '33F', '36F', '18M', '25M', '28M']
extension = 'jpg'

for arg in sys.argv:
    if 'path=' in arg:
        dataset_path = arg[5:]
    if 'set=' in arg:
        subsets_to_include = arg[4:]
    if 'mode=' in arg:
        truth = arg[5:]
    if 'skip_ph=' in arg or 'skip_vi=' in arg:
        phonemes_to_exclude.append(arg[8:])
    if 'skip_speaker=' in arg:
        speakers_to_exclude.append(arg[13:])
    if 'select=' in arg:
        speaker_selection.append(arg[7:])
    if '-d' in arg:
        decimate = True
    if '-roi' in arg:
        roi = True
    if 'export=' in arg:
        export_name = arg[7:]
    if 'extension=' in arg:
        extension = arg[10:]
    
print('Creating split files...', flush=True)

trainset_name = export_name + '_train.json'
testset_name = export_name + '_test.json'
validationset_name = export_name + '_validation.json'
export_name = export_name + '.json'

if subsets_to_include is "all":
    subsets_to_include = ["volunteers", "lipspeakers"]
else:
    subsets_to_include = [subsets_to_include]

print(subsets_to_include, flush=True)
if len(speakers_to_exclude) > 0:
    print('Excluding speakers', speakers_to_exclude, flush=True)
if len(phonemes_to_exclude) > 0:
    print('Excluding phonemes/visemes', phonemes_to_exclude, flush=True)

# create dictionary that will then be exported as a json file
# structure: [speaker][sequence][frame] = (path, truth)
dataset = {}
for subset in subsets_to_include:
    # get a list of all speakers in the subset
    if len(speaker_selection) == 0:
        speaker_paths = [name for name in os.listdir(os.path.join(dataset_path, subset))        # Get all speakers available
                        if os.path.isdir(os.path.join(dataset_path, subset, name))
                        and name not in speakers_to_exclude]
    else:
        speaker_paths = [name for name in speaker_selection
                        if os.path.isdir(os.path.join(dataset_path, subset, name))]
    for speaker in speaker_paths:
        dataset[speaker] = {}
        # get a list of all sequences for the speaker
        sequence_paths = [name for name in os.listdir(os.path.join(dataset_path, subset, speaker, 'Clips', 'straightcam'))
                          if os.path.isdir(os.path.join(dataset_path, subset, speaker, 'Clips', 'straightcam', name))]
        for sequence in sequence_paths:
            dataset[speaker][sequence] = {}
            # get a list of all frames of the sequence
            frames = [name for name in os.listdir(os.path.join(dataset_path, subset, speaker, 'Clips', 'straightcam', sequence))
                      if os.path.isfile(os.path.join(dataset_path, subset, speaker, 'Clips', 'straightcam', sequence, name)) and (name.endswith(extension))]
            # get a list of all visemes/phonemes of the sequence. skip if no list has been created.
            truth_path = os.path.join(dataset_path, subset, speaker, 'Clips', 'straightcam', sequence, truth + '.txt')
            if not os.path.exists(truth_path):
                print('No phoneme/viseme file for ' + speaker + ' ' + sequence + '. Skipping this sequence.', flush=True)
                continue
            truths = {}
            with open(truth_path) as file:
                line = file.readline()
                while line:
                    # skip commented lines
                    if line.startswith('#'):
                        line = file.readline()
                        continue

                    f, t, _ = line.split(' ')
                    truths[int(f)] = t
                    line = file.readline()
                file.close()

            # write paths and ground truths to the dictionary
            for i in truths.keys():
                if truths[i] not in phonemes_to_exclude:
                    if not roi:
                        frame_path = os.path.join(dataset_path, subset, speaker, 'Clips', 'straightcam', sequence, str(i) + '.jpg')
                    else:
                        frame_path = os.path.join(dataset_path, subset, speaker, 'Clips', 'ROI', sequence, str(i) + '.jpg')
                    dataset[speaker][sequence][i] = (frame_path, truths[i])

            if decimate:
                start_frame = 1
                end_frame = -1
                current_class = dataset[speaker][sequence][1][1]
                decimated_sequence = {}
                for i in dataset[speaker][sequence].keys():
                    if dataset[speaker][sequence][i][1] != current_class:
                        # class change!
                        end_frame = i - 1
                        middle_frame = start_frame + int((end_frame - start_frame) / 2)
                        decimated_sequence[middle_frame] = dataset[speaker][sequence][middle_frame]
                        current_class = dataset[speaker][sequence][i][1]
                        start_frame = i
                dataset[speaker][sequence] = decimated_sequence


# export the full dataset as a json file
app_json = json.dump(dataset, open(export_name, 'w'))
print('Dataset file exported to ' + export_name, flush=True)

if create_traintestval_split:
    print('Creating train-test-split according to recommended split...', flush=True)

    # Create a test set from the recommended split
    traindataset = {}
    testset = {}
    for speaker in dataset:
        if speaker in testset_speakers:
            testset[speaker] = dataset[speaker]
        else:
            traindataset[speaker] = dataset[speaker]
        
    # Split the trainset again into train set and validation set (75:25)
    trainset = {}
    validationset = {}
    
    #trainspeakers = [key for key in traindataset.keys()]
    #trainspeakers = random.choices(trainspeakers, k=int(0.75 * len(traindataset.keys())))    # 75 % of keys in traindataset at random

    for speaker in traindataset:
        if speaker in validationset_speakers:
            validationset[speaker] = traindataset[speaker]
        else:
            trainset[speaker] = traindataset[speaker]

    app_json = json.dump(trainset, open(trainset_name, 'w'))
    print('Trainset file exported to ', trainset_name)
    print('Speakers in trainset: ', trainset.keys(), '\n')

    app_json = json.dump(validationset, open(validationset_name, 'w'))
    print('Validationset file exported to ', validationset_name)
    print('Speakers in validationset: ', validationset.keys(), '\n')

    app_json = json.dump(testset, open(testset_name, 'w'))
    print('Testset file exported to ', testset_name)
    print('Speakers in testset: ', testset.keys(), '\n')

print('All done.', flush=True)