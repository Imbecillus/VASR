import os
import os.path as osp
import imageio
from PIL import Image
import torch
import numpy as np

directory = '/beegfs/work/shared/TCD-TIMIT/volunteers/'

print('Converting and rescaling everything from', directory, 'to 100x100 pixel JPG and PT files.')

people = [name for name in os.listdir(directory) if osp.isdir(osp.join(directory, name))]

for person in people:
    person_path = osp.join(directory, person, 'Clips', 'ROI')
    print(person)

    clips = [name for name in os.listdir(person_path) if osp.isdir(osp.join(person_path, name))]
    
    for clip in clips:
        print('  Converting', clip, flush=True)
        images = [name for name in os.listdir(osp.join(person_path, clip)) if osp.isfile(osp.join(person_path, clip, name)) and name.endswith('.jpg')]
        for imagepath in images:
            imagepath = osp.join(person_path, clip, imagepath)

            image = imageio.imread(imagepath)                   
            image = Image.fromarray(image).resize((100,100))

            image.save(imagepath)                               # Save image as JPG

            image = np.array(image)                             # Save image as PT-file
            # image = image / 255   # moved to dataset class, because this would blow up the file size a LOT
            image = np.transpose(image, [2, 0, 1])
            image = torch.from_numpy(image)
            imagepath = imagepath[:-4] + '.pt'
            torch.save(image, imagepath)

print('Done.\n\n')