# IMPORTS
import sys
import os
import torch
import numpy as np
import torchvision.transforms
from scipy.fftpack import dct
from scipy.fftpack import idct
from PIL import Image
import imageio
from zigzag import zigzag
from zigzag import inverse_zigzag

##################
# READ ARGUMENTS #
##################
seq_dir = sys.argv[1]
n_feats = int(sys.argv[2])
verbose = True if '-v' in sys.argv else False
if '-d' in sys.argv:
    deltas = 1
elif '-dd' in sys.argv:
    deltas = 2
else:
    deltas = 0

#############
# FUNCTIONS #
#############
def load_img(path):
    """
    Load image, resize to 24x16, convert to grayscale and remove redundant dimensions
    """
    img = imageio.imread(path)
    img = Image.fromarray(img)
    t = torchvision.transforms.Compose([torchvision.transforms.Resize((16,24)), torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])
    img = t(img)[0,:,:]
    img = np.array(img)
    return img

def DCT_2d(img):
    img = dct(img, type=2, norm='ortho', axis=1)
    img = dct(img, type=2, norm='ortho', axis=0)
    return img

def IDCT_2d(img):
    img = idct(img, type=2, norm='ortho', axis=0)
    img = idct(img, type=2, norm='ortho', axis=1)
    return img

def save(feats, path):
    path = path[0:-4] + '_feats.pt'
    torch.save(feats, path)

def calc_deltas(sequence):
    deltas = []

    deltas.append(sequence[0] - sequence[0])

    for i in range(1,len(sequence)):
        if verbose:
            if i % int(0.1 * len(sequence) + 1) == 0:
                print('#', end='')
        delta = sequence[i] - sequence[i - 1]
        deltas.append(delta)
    if verbose:
        print(' done.')

    return deltas

###########
# PROGRAM #
###########
if verbose:
    print(f'Path: {seq_dir}; extracting {n_feats} DCT features per frame.')

# Get list of image frames
frames = [f for f in os.listdir(seq_dir)
          if f.endswith('.jpg') and os.path.isfile(os.path.join(seq_dir, f))]
if verbose:
    print(f'{len(frames)} frames.')

# Extract DCT features for each frame
i = 0
seq_feats = []
if verbose:
    print('DCT Feats:   ', end='')
for frame in frames:
    framepath = os.path.join(seq_dir, frame)
    img = load_img(framepath)               # Load image
    dct_matrix = DCT_2d(img)                # Perform 2D-DCT
    frame_feats = zigzag(dct_matrix)        # Convert matrix to vector
    frame_feats = frame_feats[1:n_feats+1]  # Retain only n_feats coefficients (and skip the first one)
    seq_feats.append(frame_feats)           # Append to list of all frame features
    i += 1
    if verbose:
        if i % int(0.1 * len(frames) + 1) == 0:
            print('#', end='')

if verbose:
    print(' done.')

# Calculate Deltas
if deltas >= 1:
    if verbose:
        print('Deltas:      ', end='')
    seq_deltas = calc_deltas(seq_feats)

# Calculate Delta-deltas
if deltas >= 2:
    if verbose:
        print('Deltadeltas: ', end='')
    seq_deltadeltas = calc_deltas(seq_deltas)

# Save
i = 0
for frame in frames:
    framepath = os.path.join(seq_dir, frame)
    framefeats = seq_feats[i].tolist()
    if deltas >= 1:
        framefeats = framefeats + seq_deltas[i].tolist()
    if deltas >= 2:
        framefeats = framefeats + seq_deltadeltas[i].tolist()
    save(framefeats, framepath)
    i += 1

if verbose:
    print('Features generated and saved.')