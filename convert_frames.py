import os
import os.path as osp
import imageio
from PIL import Image
import torch
import numpy as np

directory = '/beegfs/work/shared/TCD-TIMIT/volunteers/'
ignore = ['01M', '02M', '03F', '04M', '05F', '06M', '07F', '08F', '09F', '10M', '11F', '12M', '13F', '14M', '15F', '16M', '17F', '18M', '19M', '20M', '21M', '22M', '23M', '24M', '25M', '26M', '27M', '28M', '29M']

print('Converting and rescaling everything from', directory, 'to 36x36 pixel JPG and PT files.')

people = [name for name in os.listdir(directory) if osp.isdir(osp.join(directory, name))]


for person in people:
    if person in ignore:
        print(f'Ignoring {person}')
        continue

    person_path = osp.join(directory, person, 'Clips', 'ROI')
    print(person)

    clips = [name for name in os.listdir(person_path) if osp.isdir(osp.join(person_path, name))]
    
    for clip in clips:
        print('  Converting', clip, flush=True)
        images = [name for name in os.listdir(osp.join(person_path, clip)) if osp.isfile(osp.join(person_path, clip, name)) and name.endswith('.jpg')]
        for imagepath in images:
            imagepath = osp.join(person_path, clip, imagepath)

            image = imageio.imread(imagepath)                   
            image = Image.fromarray(image).resize((36,36))

            # image.save(imagepath)                               # Save image as JPG

            image = np.array(image)                             # Save image as PT-file
            # image = image / 255   # moved to dataset class, because this would blow up the file size a LOT
            image = np.transpose(image, [2, 0, 1])
            image = torch.from_numpy(image)
            imagepath = imagepath[:-4] + '.pt36'
            torch.save(image, imagepath)

print('Done.\n\n')