# Imports
import sys
import os
import torch
import torchvision
import numpy as np
import imageio
import PIL
from PIL import Image
from torch import nn

# Parsing command line
export = False
posteriors = False
config_path = None
path = None
map = None

for arg in sys.argv:
    if arg.startswith('--conf='):
        config_path = arg[7:]
    elif arg.startswith('--path='):
        path = arg[7:]
    elif arg.startswith('--export='):
        export_path = arg[9:]
        open(export_path, 'w').write(f'%{path}\n')
        export = True
    elif arg.startswith('--map='):
        map = arg[6:]
    elif arg.startswith('--post'):
        posteriors = True

assert map in [None, 'jeffers-to-phn', 'lee-to-phn'], 'Unknown mapping specified.'

arg_string = ""
if os.path.isfile(config_path):
    with open(config_path) as config_lines:
        line = config_lines.readline()
        while line:
            arg_string += " " + line
            line = config_lines.readline()
else:
    print(f'Error: {config_path} is not a file.')
    exit()

# Functions
def load_model():
    transforms = []

    if torch.cuda.is_available():
        print("CUDA available", flush=True)
        device = torch.device("cuda")
    else:
        print("CUDA not available", flush=True)
        device = torch.device("cpu")

    with open(config_path) as config:
        import viseme_list

        context = 0
        channels = 1

        line = config.readline()
        while line:
            if 'model=' in line:
                choose_model = line[6:].replace('\n', '')
            if 'import=' in line:
                savepath = line[7:].replace('\n', '')
            if 'visemes=' in line:
                viseme_set = line[8:].replace('\n', '')
                if viseme_set == 'jeffersbarley':
                    t_t = viseme_list.visemes_jeffersbarley
                elif viseme_set == 'lee':
                    t_t = viseme_list.visemes_lee
                elif viseme_set == 'neti':
                    t_t = viseme_list.visemes_neti
                else:
                    t_t = viseme_list.phonemes
            if 'color=' in line:
                if 'true' in line:
                    channels = 3
                    transforms = [
                        torchvision.transforms.Resize((36,36))
                    ]
                else:
                    channels = 1
                    transforms = [
                            torchvision.transforms.Grayscale(),
                            torchvision.transforms.Resize((36, 36))
                        ]
            if 'context=' in line:
                context = int(line[8:].replace('\n', ''))
            line = config.readline()

    print(f'Model={choose_model}; visemes={viseme_set}; channels={channels}; importing {savepath}')

    model = None

    if choose_model == 'simple_CNN':
        from architectures import simple_CNN as architecture
    elif choose_model == 'ConvNet':
        from architectures import lipreading_in_the_wild_convnet as architecture
    elif choose_model == 'UnnConvNet':
        from architectures import lipreading_in_the_wild_convnet_unnormalized as architecture
    elif choose_model == 'RNN-ConvNet':
        from architectures import rnn_convnet as architecture
    elif choose_model == 'ResNet10':
        from architectures import sigmedia as architecture
        model = architecture.Net(channels * (2 * context + 1), len(t_t), 128, (8, 16, 24, 32), 0.0, device).to(device)
    elif choose_model == 'ResNet10_old':
        from architectures import sigmedia_old as architecture
        model = architecture.Net(channels * (2 * context + 1), len(t_t), 128, (8, 16, 24, 32), 0.0, device).to(device)

    assert os.path.exists(savepath), 'Specified model file does not exist.'

    if not model:
        model = architecture.Net(channels * (2 * context + 1), len(t_t))
    model.load_state_dict(torch.load(savepath, map_location=device))

    transforms = list(transforms)
    transforms.append(torchvision.transforms.ToTensor())
    transforms = torchvision.transforms.Compose(transforms)

    return model, transforms, t_t

def load_image(path):
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
        image = transforms(image)             # Apply transforms
        image = np.array(image)                         # Convert image into numpy array
        
    return image

def predict(path):
    image = load_image(path)
    xb = torch.from_numpy(image)
    shape = xb.shape
    xb = xb.reshape((1, shape[0], shape[1], shape[2]))

    prediction = model(xb)
    _, p = torch.max(prediction, 1)

    return truth_table[p]

def posts_from_image(image):
    xb = torch.from_numpy(image)
    shape = xb.shape
    xb = xb.reshape((1, shape[0], shape[1], shape[2]))

    return model(xb)

def posts(path):
    image = load_image(path)
    
    return posts_from_image(image)

def print_prediction(path, frame, prediction):
    prediction = prediction.view(len(prediction[0]))

    if posteriors and map is not None:
        # Perform mapping
        if map == 'jeffers-to-phn':
            from viseme_list import visemes_jeffersbarley
            from viseme_list import jeffersbarley_to_phonemes
            from viseme_list import phonemes

            phoneme_posteriors = {}
            for i in range(len(visemes_jeffersbarley)):
                for phoneme in jeffersbarley_to_phonemes[visemes_jeffersbarley[i]]:
                    phoneme_posteriors[phoneme] = prediction[i]
            
            new_prediction = [None] * len(phonemes)
            for i in range(len(phonemes)):
                new_prediction[i] = phoneme_posteriors.get(phonemes[i], 0)

        elif map == 'lee-to-phn':
            from viseme_list import visemes_lee
            from viseme_list import lee_to_phonemes
            from viseme_list import phonemes

            phoneme_posteriors = {}
            for i in range(len(visemes_lee)):
                for phoneme in lee_to_phonemes[visemes_lee[i]]:
                    phoneme_posteriors[phoneme] = prediction[i]
            
            new_prediction = [None] * len(phonemes)
            for i in range(len(phonemes)):
                new_prediction[i] = phoneme_posteriors.get(phonemes[i], 0)

        prediction = new_prediction


    if not frame:
        out_str = f"{path}: {prediction}"
    else:
        out_str = f"{path}_{frame}: {prediction}"

    if not export:
        print(out_str)
    else:
        lines = open(export_path).readlines()
        lines.append(out_str + "\n")
        open(export_path, 'w').writelines(lines)

# Runtime
model, transforms, truth_table = load_model()
if os.path.isfile(path):
    print(f'Generating prediction for frame at {path}.')
    if posteriors:
        prediction = posts(path)
    else:
        prediction = predict(path)
    print_prediction(path, None, prediction)
elif os.path.isdir(path):
    print(f'Generating predictions for sequence at {path}.')
    n_frames = len([name for name in os.listdir(path) if name.endswith('.pt')])
    for i in range(n_frames):
        if posteriors:
            prediction = posts(os.path.join(path, str(i + 1) + '.pt'))
        else:
            prediction = predict(os.path.join(path, str(i + 1) + '.pt'))
        print_prediction(path, i + 1, prediction)
else:
    print(f'Error: {path} does not exist.')
    exit()

print('Done.')