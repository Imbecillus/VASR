import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import sys
import json
import numpy as np
from scipy import misc
import imageio
import viseme_list
import PIL
from PIL import Image

class TCDTIMITDataset(Dataset):
    """ Class for the TCD-TIMIT corpus """

    def __init__(self,
                 dataset,
                 data_transforms=None,
                 n_files=None,
                 viseme_set=None,
                 sequences=False,
                 truth='one-hot',
                 context=0):

        """
        Initialises the dataset by loading the desired data

        :param dataset: path to the dataset json
        :param data_transforms: takes the transforms.compose list
        :param n_files: How many files shall be loaded. Files are selected randomly if there are more files than n_files
                        Seeded by numpy.random.seed()
        :param viseme_set: What viseme set is being used? If unspecified, assuming phonemes
        :param sequences: Use sequences or individual frames as items returned by getitem? If True, getitem will return a list of image-truth-pairs belonging to a sequence. Defaults to False.
        :param truth: 'one-hot'|'index'. Can be used to specify whether the ground truth shall be presented as a one-hot vector or as a index.
        :param context: Specify how many frames of context shall be provided along with each frame
        """

        super(TCDTIMITDataset, self).__init__()
        assert isinstance(dataset, str)
        
        if data_transforms is None:
            data_transforms = []
        
        self.dataset = dataset
        self.data_transforms = list(data_transforms)
        self.data_transforms.append(transforms.ToTensor())
        self.data_transforms = transforms.Compose(self.data_transforms)
        self.sequences = sequences
        self.truth = truth
        self.context = context
        if viseme_set == None:
            self.truth_table = viseme_list.phonemes
        else:
            assert viseme_set in ('neti', 'jeffersbarley', 'lee'), '''currently supported viseme sets are Neti, Jeffers-Barley and Lee'''
            if viseme_set == 'neti':
                self.truth_table = viseme_list.visemes_neti
            elif viseme_set == 'jeffersbarley':
                self.truth_table = viseme_list.visemes_jeffersbarley
            else:
                self.truth_table = viseme_list.visemes_lee

        # create data structure for __getitem__ which contains paths to image files
        json_data = json.load(open(dataset, 'r'))
        data = []
        self.seq_starts = []
        # iterate through speakers
        for speaker_key in json_data:
            # iterate through sequences
            for seq_key in json_data[speaker_key]:
                self.seq_starts.append(len(data))                # Create a list of the start frame indices for each sequence
                if not sequences:
                    # iterate through frames and add frames to data individually
                    for frame_number in json_data[speaker_key][seq_key]:
                        pair = json_data[speaker_key][seq_key][frame_number]
                        path = pair[0].replace('/', os.sep)
                        path = path.replace('\\', os.sep)
                        if os.path.exists(path):
                            data.append(json_data[speaker_key][seq_key][frame_number])
                else:
                    # add frames to data as a list
                    sequence_data = []

                    for frame_number in json_data[speaker_key][seq_key]:
                        pair = json_data[speaker_key][seq_key][frame_number]
                        path = pair[0].replace('/', os.sep)
                        path = path.replace('\\', os.sep)
                        if os.path.exists(path):
                            sequence_data.append(json_data[speaker_key][seq_key][frame_number])

                    data.append(sequence_data)
        
        # randomly select n_files frames to keep in data
        if n_files is not None:
            # if we have less frames than n_files, we don't need to do anything
            if n_files < len(data):
                indices_to_keep = np.random.choice(len(data), n_files, replace=False)
                data = [data[i] for i in indices_to_keep]

        self.data = data

    def __get_sequence_start_stop(self, number):
        """
        Returns the first and last frame indices of the sequence the given frame is part of.
        """
        start = 0
        stop = 0
        
        for seq_id in self.seq_starts:
            if number >= seq_id:
                start = seq_id
            else:
                stop = seq_id - 1
                break

        if stop == 0:
            stop = len(self.data) - 1

        return start, stop

    def __len__(self):
        """Returns the number of elements inside the dataset."""
        return len(self.data)

    def __loadimage(self, path):
        """
        Loads image from the specified path into a numpy array. Path can be to an image file supported by imageio or a pt-file containing a torch tensor.
        """
        path = path.replace('/', os.sep)                    # Set system-appropriate path seperator in path
        path = path.replace('\\', os.sep)

        if path.endswith('.pt'):
            image = torch.load(path)
            image = np.array(image)
            image = image / 255

        else:
            image = imageio.imread(path)                    # Read image into PIL format
            image = Image.fromarray(image)
            image = self.data_transforms(image)             # Apply transforms
            image = np.array(image)                         # Convert image into numpy array

        return image

    def __get_image_in_sequence(self, n, start, stop):
        """
        Checks whether index n is between start and stop. Returns image at n if yes; returns start- or stop-frame as appropriate if no.
        """
        if n < start:
            path, _ = self.data[start]
            return self.__loadimage(path)

        if n > stop:
            path, _ = self.data[stop]
            return self.__loadimage(path)
            
        path, _ = self.data[n]
        return self.__loadimage(path)

    def __get_image_context_and_truth(self, number):
        """
        Gets truth_tensor for the specified image and appends context frames, if requested.
        """

        path, truth = self.data[number]

        if self.truth == 'one-hot':
            truth_tensor = torch.zeros(len(self.truth_table))           # Create a n-dimensional tensor...
            truth_tensor[self.truth_table.index(truth)] = 1             # ...and set index of ground truth to 1.
        elif self.truth == 'index':
            truth_tensor = torch.tensor(self.truth_table.index(truth), dtype=torch.long)    # Create a tensor containing only the index of the ground truth.

        if self.context is not 0:
            # add context around the frame as additional channels
            numbers = range(number - self.context, number + self.context + 1)

            # get the bounds of the current sequence:
            start, stop = self.__get_sequence_start_stop(number)

            # read the individual frames in
            frames = {}
            for n in numbers:
                frames[n] = self.__get_image_in_sequence(n, start, stop)
                                        
            n_start = number - self.context                             # index of first context frame
            shape = frames[number].shape                                # save shape for the multi-channel image and padding frames
            
            # Create image with the regular dimensions but additional channels for the context in both directions
            image = np.empty([(1 + 2 * self.context) * shape[0], shape[1], shape[2]])

            for x in range(shape[1]):
                for y in range(shape[2]):
                    for n in range(1 + 2 * self.context):
                        for c in range(shape[0]):
                            image[n + c][x][y] = frames[n_start + n][c][x][y]
        else:
            image = self.__loadimage(path)

        return image, truth_tensor

    def __getitem__(self, number):
        """
        Loads dataset element with index number 'number'.
        """

        if self.sequences:
            sequence = []
            for path, truth in self.data[number]:
                image = self.__loadimage(path)
                if self.truth == 'one-hot':
                    truth_tensor = torch.zeros(len(self.truth_table))           # Create a n-dimensional tensor...
                    truth_tensor[self.truth_table.index(truth)] = 1             # ...and set index of ground truth to 1.
                elif self.truth == 'index':
                    truth_tensor = torch.tensor(self.truth_table.index(truth), dtype=torch.long)    # Create a tensor containing only the index of the ground truth.

                sequence.append([image, truth_tensor])                  # Add image-truth-pair to sequence list
            return sequence
        else:
            return self.__get_image_context_and_truth(number)