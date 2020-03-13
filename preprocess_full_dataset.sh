#!/bin/bash
#SBATCH --job-name=preprocess_TCDTIMIT
#SBATCH --output=create_json_files.out
#SBATCH --time=0-01:30:00
#SBATCH --mail-user=timbecillus@gmail.com
#SBATCH --mail-type=ALL
python create_dataset_json.py set=volunteers mode=visemes path=/beegfs/work/shared/TCD-TIMIT export=splits/full_vis_jpg extension=jpg \
    skip_speaker=35M skip_speaker=53M skip_speaker=27M
python create_dataset_json.py set=volunteers mode=visemes path=/beegfs/work/shared/TCD-TIMIT export=splits/roi_vis_jpg extension=jpg -roi \
    skip_speaker=35M skip_speaker=53M skip_speaker=27M
python create_dataset_json.py set=volunteers mode=visemes path=/beegfs/work/shared/TCD-TIMIT export=splits/full_vis_pt extension=pt \
    skip_speaker=35M skip_speaker=53M skip_speaker=27M
python create_dataset_json.py set=volunteers mode=visemes path=/beegfs/work/shared/TCD-TIMIT export=splits/roi_vis_pt extension=pt -roi \
    skip_speaker=35M skip_speaker=53M skip_speaker=27M

python create_dataset_json.py set=volunteers mode=phonemes path=/beegfs/work/shared/TCD-TIMIT export=splits/full_pho_jpg extension=jpg \
    skip_speaker=35M skip_speaker=53M skip_speaker=27M
python create_dataset_json.py set=volunteers mode=phonemes path=/beegfs/work/shared/TCD-TIMIT export=splits/roi_pho_jpg extension=jpg -roi \
    skip_speaker=35M skip_speaker=53M skip_speaker=27M
python create_dataset_json.py set=volunteers mode=phonemes path=/beegfs/work/shared/TCD-TIMIT export=splits/full_pho_pt extension=pt \
    skip_speaker=35M skip_speaker=53M skip_speaker=27M
python create_dataset_json.py set=volunteers mode=phonemes path=/beegfs/work/shared/TCD-TIMIT export=splits/roi_pho_pt extension=pt -roi \
    skip_speaker=35M skip_speaker=53M skip_speaker=27M