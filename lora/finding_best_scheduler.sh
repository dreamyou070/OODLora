#!/bin/bash

#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos big_qos
#SBATCH --job-name=parksooyeon_finding_best_scheduler
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:2
#SBATCH --output=./result/printing_screen/finding_best_scheduler.log
#--nodes=1 --ntasks-per-node 1
# sbatch -q big_qos --nodes 2 --output=../result/printing_screen/2_contrastive_learning_eps_0.0_new_code_highrepeat_test.txt training_contrastive.sh
# run on back = 2>&1
# sbatch -p suma_rtx4090 -q big_qos --nodes 1 --output=../result/printing_screen/noising_scheduler_log.log finding_best_scheduler.sh 2>&1

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv
ml purge
ml load cuda/11.0
