#!/bin/bash
#SBATCH --job-name=train_Lee_tcdtimit
#SBATCH --output=200127_convnet_CEL_full_ROI_Lee.out
#SBATCH --partition=gpu
#SBATCH --time=8-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=timbecillus@gmail.com
#SBATCH --mail-type=ALL
python train.py \
    epochs=50 \
    model=UnnConvNet \
    visemes=lee \
    color=true \
    dataset=splits/roi_Lee_pt_train.json \
    batch_size=256 \
    dropout=0.5 \
    weighted_loss=true \
    learning_rate=0.00001 \
    export=results/200127_convnet_CEL_full_ROI_Lee_dr50.pth -i10
python eval.py \
    model=ConvNet \
    color=true \
    visemes=lee \
    validationset=splits/roi_Lee_pt_validation.json  \
    import=results/200120_convnet_CEL_full_ROI_Lee_dr50.pth