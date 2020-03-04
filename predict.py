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
config_path = sys.argv[1]
path = sys.argv[2]
if len(sys.argv) > 3:
    export_path = sys.argv[3]
    export = True
else:
    export = False

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

        line = config.readline()
        while line:
            if 'model=' in line:
                choose_model = line[6:-1]
            if 'import=' in line:
                savepath = line[7:-1]
            if 'visemes=' in line:
                viseme_set = line[8:-1]
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
                        torchvision.transforms.Resize((100,100))
                    ]
                else:
                    channels = 1
                    transforms = [
                            torchvision.transforms.Grayscale(),
                            torchvision.transforms.Resize((100, 100))
                        ]
            if 'context=' in line:
                context = int(line[8:-1])
            line = config.readline()

    if choose_model == 'simple_CNN':
        from architectures import simple_CNN as architecture
    elif choose_model == 'ConvNet':
        from architectures import lipreading_in_the_wild_convnet as architecture
    elif choose_model == 'UnnConvNet':
        from architectures import lipreading_in_the_wild_convnet_unnormalized as architecture
    elif choose_model == 'RNN-ConvNet':
        from architectures import rnn_convnet as architecture
    elif choose_model == 'ResNet18':
        transforms.append(torchvision.transforms.Resize((256,256)))
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, len(truth_table))
    elif choose_model == 'DrResNet18':
        transforms.append(torchvision.transforms.Resize((256,256)))
        from architectures import ResNet18_dropout as architecture
    elif choose_model == 'DrResNet18b':
        transforms.append(transforms.Resize((256,256)))
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Dropout(0), nn.Linear(512, len(truth_table)))

    assert os.path.exists(savepath), 'Specified model file does not exist.'

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

def print_prediction(path, frame, prediction):
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
    prediction = predict(path)
    print_prediction(path, None, prediction)
elif os.path.isdir(path):
    print(f'Generating predictions for sequence at {path}.')
    n_frames = len([name for name in os.listdir(path) if name.endswith('.pt')])
    for i in range(n_frames):
        prediction = predict(os.path.join(path, str(i + 1) + '.pt'))
        print_prediction(path, i + 1, prediction)
else:
    print(f'Error: {path} does not exist.')
    exit()

print('Done.')