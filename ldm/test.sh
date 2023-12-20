#!/bin/bash
#SBATCH --partition=compute-od-gpu
#SBATCH --job-name=intelmpi_test
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --comment "Key=Monitoring,Value=ON"
#SBATCH --exclusive

# sbatch -q big_qos --nodes 2 --output=../result/printing_screen/2_contrastive_learning_eps_0.0_new_code_highrepeat_test.txt training_contrastive.sh
# sbatch -p big_suma_rtx3090 --qos big_qos --ntasks-per-node 1 --output /home/dreamyou070/Lora/OODLora/result/printing_screen/test_log.log   --cpus-per-gpu=6 train_contrastive_2.sh


# 1) gpu config : nprocess 2

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

python scripts/knn2img.py  --prompt "a happy bear reading a newspaper, oil on canvas"