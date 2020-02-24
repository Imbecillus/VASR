import os
import sys
import subprocess
import viseme_list
import imageio
import mouth_detection
import numpy as np
import torch
from PIL import Image

# PARAMETERS
dataset_path = "data"           # path to the TCD-TIMIT data
processing_mode = "volunteers"  # modes: all, lipspeakers, volunteers
viseme_mode = "jeffers"         # viseme map to be used: jeffers, neti, lee or None
skip_person = []
extract = False
crop = False
crop_individually = False
verbose = False

for arg in sys.argv:
    if 'set=' in arg:
        processing_mode = arg[4:]
    if 'mode=' in arg:
        viseme_mode = arg[5:]
    if 'path=' in arg:
        dataset_path = arg[5:]
    if '-e' in arg:
        extract = True
    if '-c' in arg:
        crop = True
    if '-i' in arg:
        crop_individually = True
    if '-v' in arg or '--verbose' in arg:
        verbose = True

def MP4_to_images(filepath):
    """
    Converts the frames of a video file to individual jpg files in a new folder named the same as the video file.

    :param filepath: path to the video file to be converted
    """

    # Create a new directory for the seperate frames
    video_directory = filepath[0:-4]
    if not os.path.isdir(video_directory):
        os.mkdir(video_directory)

    # Use ffmpeg to extract all frames of the video to seperate jpg files
    command = ['ffmpeg',
		   '-i', filepath,
		   '-s', '1000x1000',
		   '-vf', 'crop=1000:1000:460:40',
           '-qscale:v', '4',
           '-loglevel', 'error',
		   os.path.join(video_directory, "%d.jpg")]

    p = subprocess.Popen(command)
    subprocess.Popen.wait(p)

def create_phoneme_table(video_directory, speaker_id, clip_id, global_phoneme_table, mode='None'):
    """
    Creates a table that shows the currently spoken phoneme for each frame

    :param video_directory: path to the directory for the phoneme table to be saved in
    :param speaker_id: ID of the speaker in the clip
    :param clip_id: ID of the clip for which the phoneme table is to be saved
    :param global_phoneme_table: the global phoneme table created by read_label_file(filepath)
    :param mode: declares whether an additional viseme table shall be created and what map shall be used. can be 'neti', 'jeffers' or 'lee'. standard is phonemes only
    """
    
    # Check if clip_id is actually contained in the phoneme table
    if clip_id not in global_phoneme_table[speaker_id].keys():
        return False
    
    framelength = 10000000 / 29.97
    frames = [name for name in os.listdir(video_directory)
              if os.path.isfile(os.path.join(video_directory, name))
              and name.endswith('jpg')]
    phoneme_table = {}

    for i in range(len(frames)):
        # find phoneme timeslot that frame i falls into
        for time in global_phoneme_table[speaker_id][clip_id].keys():
            if i * framelength >= int(time):
                phoneme_table[i + 1] = global_phoneme_table[speaker_id][clip_id][time]

    # save phoneme table to file
    filepath = os.path.join(video_directory, 'phonemes.txt')
    with open(filepath, 'w') as file:
        for frame in phoneme_table.keys():
            file.writelines(str(frame) + ' ' + phoneme_table[frame] + ' ' + str((frame - 1) * framelength) + '\n')
        file.close()

    if mode != 'None':
        filepath = os.path.join(video_directory, 'visemes.txt')
        viseme_map = {}

        if mode == 'neti':
            viseme_map = viseme_list.phonemes_to_neti
        elif mode == 'lee':
            viseme_map = viseme_list.phonemes_to_lee
        else:
            viseme_map = viseme_list.phonemes_to_jeffersbarley

        with open(filepath, 'w') as file:
            file.writelines('# VISEME MAPPING: ' + mode + '\n')
            for frame in phoneme_table.keys():
                file.writelines(str(frame) + ' ' + get_viseme(phoneme_table, frame, viseme_map) + ' ' + str((frame - 1) * framelength) + '\n')

def get_viseme(phoneme_table, frame, map):
    # Some phonemes are missing in some viseme maps.
    # The reason for this is that phonemes like /hh/ and /hv/ do not have visual features associated with them. (see Gillen thesis provided with TCD-TIMIT)
    # Gillen proposes mapping these phonemes to the next viseme in the sequence.

    viseme = map.get(phoneme_table[frame], 'missing')
    if viseme is 'missing':
        viseme = get_viseme(phoneme_table, frame + 1, map)
    
    return viseme

def read_label_file(filepath):
    """
    Creates a global phoneme dictionary based on the MLF files provided with TCD-TIMIT
    phoneme_table will be a dictionary of dictionaries of dictionaries:
    PHONEME_TABLE
        - Speaker 1:
            - Clip 1:
                - Time : Phoneme
                - Time : Phoneme
                - ...
            - ...
        - ...

    :param filepath: path to the MLF file from which the table is to be created
    """
    phoneme_table = {}
    with open(filepath, 'r') as file:
        line = file.readline()
        speaker_id = ""
        clip_id = ""

        while line:
            if line.startswith('#') or line.startswith('.'):            # skip comments and dot lines
                line = file.readline()
                continue
            
            line = line[0:-1]                                           # remove newline character

            if line.startswith('"'):
                # Line has the format: "/path/to/TCDTIMIT/[speakerset]/[speakerid]/Clips/straightcam/[clipid].mp4"
                # Get speaker id and clip id
                line = line.strip('"')
                split = line.split('/')
                speaker_id = split[-4]
                clip_id = split[-1]
                clip_id = clip_id[0:len(clip_id) - 4]                   # remove file extension from clip_id

                # if speaker_id does not already have a dictionary, create one
                if not speaker_id in phoneme_table:
                    phoneme_table[speaker_id] = {}

                # add clip dictionary to speaker dictionary
                phoneme_table[speaker_id][clip_id] = {}
                line = file.readline()
                continue

            # Line is not a comment, file name or dot. Add phoneme to table
            start_time, end_time, phoneme = line.split(' ')
            
            phoneme_table[speaker_id][clip_id][start_time] = phoneme
            
            line = file.readline()                                      # read next line
    return phoneme_table


# RUNTIME
print("TCD-TIMIT Preprocessor")
if extract:
    print("This script will convert " + processing_mode + " clips in '" + dataset_path + "' into seperate jpg files and create corresponding phoneme tables.")
if crop:
    print("Cropped ROI frames for the clips in '" + dataset_path + "' will be created.")
if not extract and not crop:
    print("Nothing to do.")
    exit(0)
#if input("Does that sound right (y/n)? ") == 'n':
#    exit(0)

not_found = []

if processing_mode == "all" or processing_mode == "volunteers":
    print("Processing volunteers")
    volunteer_phoneme_table = read_label_file(os.path.join(dataset_path, 'volunteer_labelfiles.mlf'))
    
    volunteers_path = os.path.join(dataset_path, 'volunteers')
    # Create list of all subdirectories in volunteers folder
    print(os.listdir(volunteers_path))
    volunteers = [name for name in os.listdir(volunteers_path)
                  if os.path.isdir(os.path.join(volunteers_path, name))]

    for person in volunteers:
        if person in skip_person:
            print('Skipping', person)
            continue

        print('\nConverting clips from ' + person, flush=True)

        # List all mp4 clips for this volunteer
        person_path = os.path.join(volunteers_path, person, "Clips", "straightcam")
        clips = [name for name in os.listdir(person_path)
                 if os.path.isfile(os.path.join(person_path, name))
                 and name.endswith('mp4')]

        for clip in clips:
            clip = clip[0:-4]

            if extract:
                # Convert clip into seperate frame images
                print('Extracting frames for ' + clip, flush=True)
                MP4_to_images(os.path.join(person_path, clip + '.mp4'))

            if crop:
                # Create additional ROI-cropped files for each frame
                print('Creating cropped ROI images for ' + clip, flush=True)
                if not os.path.isdir(os.path.join(volunteers_path, person, 'Clips', 'ROI')):
                    os.mkdir(os.path.join(volunteers_path, person, 'Clips', 'ROI'))
                ROI_path = os.path.join(volunteers_path, person, 'Clips', 'ROI', clip)
                if not os.path.isdir(ROI_path):
                    os.mkdir(ROI_path)

                clip_path = os.path.join(person_path, clip)

                frames = [name for name in os.listdir(clip_path)
                        if os.path.isfile(os.path.join(clip_path, name))
                        and name.endswith('jpg')]

                # Find mouth region
                if crop_individually:
                    for frame in frames:
                        img = imageio.imread(os.path.join(clip_path, frame))
                        c_img, success = mouth_detection.crop_mouth(img)
                        if not success:
                            print('No mouth detected for', frame, flush=True)
                        else:
                            imageio.imwrite(os.path.join(ROI_path, frame), c_img)   # Save image as jpg file
                else:
                    # Save the first bounding box found in the clip
                    for frame in frames:
                        print('Looking for mouth in frame', frame)
                        img = imageio.imread(os.path.join(clip_path, frame))
                        m_top, m_bottom, m_left, m_right = mouth_detection.get_mouth_bounding_box(img)
                        if m_top != None and m_bottom != None and m_left != None and m_right != None:
                            break
                    
                    # Use that bounding box to create the ROIs for each frame
                    for frame in frames:
                        if verbose: 
                            print('Cropping frame', frame)
                        img = imageio.imread(os.path.join(clip_path, frame))
                        c_img, success = mouth_detection.crop_mouth(img, m_top, m_bottom, m_left, m_right)
                        if not success:
                            print('No mouth detected for', clip, flush=True)
                        else:
                            imageio.imwrite(os.path.join(ROI_path, frame), c_img)   # Save image as jpg file
            
            # create phoneme table
            print('Creating phoneme table for ' + clip, flush=True)
            if create_phoneme_table(os.path.join(person_path, clip), person, clip, volunteer_phoneme_table, viseme_mode) == False:
                not_found.append((person, clip))
                print('Clip not found in MLF:', clip, flush=True)


if processing_mode == "all" or processing_mode == "lipspeakers":
    print("Processing lipspeakers")
    lipspeaker_phoneme_table = read_label_file(os.path.join(dataset_path, 'lipspeaker_labelfiles_test.mlf'))

    lipspeakers_path = os.path.join(dataset_path, 'lipspeakers')
    # Create list of all subdirectories in lipspeakers folder
    lipspeakers = [name for name in os.listdir(lipspeakers_path)
                  if os.path.isdir(os.path.join(lipspeakers_path, name))]

    for person in lipspeakers:
        print('Converting clips from ' + person)

        # List all clips for this lipspeaker
        person_path = os.path.join(lipspeakers_path, person, "Clips", "straightcam")
        clips = [name for name in os.listdir(person_path)
                 if os.path.isfile(os.path.join(person_path, name))
                 and name.endswith('mp4')]

        for clip in clips:
            clip = clip[0:-4]
            # Convert clip into seperate frame images
            print('Extracting frames for ' + clip)
            MP4_to_images(os.path.join(person_path, clip + '.mp4'))

            # Create additional ROI-cropped files for each frame
            print('Creating cropped ROI images for ' + clip)
            if not os.path.isdir(os.path.join(lipspeakers_path, person, 'Clips', 'ROI')):
                os.mkdir(os.path.join(lipspeakers_path, person, 'Clips', 'ROI'))
            ROI_path = os.path.join(lipspeakers_path, person, 'Clips', 'ROI', clip)
            if not os.path.isdir(ROI_path):
                os.mkdir(ROI_path)

            clip_path = os.path.join(person_path, clip)

            frames = [name for name in os.listdir(clip_path)
                      if os.path.isfile(os.path.join(clip_path, name))
                      and name.endswith('jpg')]

            for frame in frames:
                img = imageio.imread(os.path.join(clip_path, frame))
                c_img = mouth_detection.crop_mouth(img)
                imageio.imwrite(os.path.join(ROI_path, frame), c_img)

            # create phoneme table
            print('Creating phoneme table for ' + clip)
            if create_phoneme_table(os.path.join(person_path, clip), person, clip, lipspeaker_phoneme_table, viseme_mode) == False:
                not_found.append((person, clip))
                print('Clip not found: ', clip)

print('The following clips were not contained in the MLF:')
print(not_found)
print('done')