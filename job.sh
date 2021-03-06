#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J My_Test
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
#BSUB -J My_Test_HPC
### specify the memory needed
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
### Number of hours needed
#BSUB -W 23:59
### added outputs and errors to files
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

echo "Runnin script..."

source venv_1/bin/activate
module load tensorrt/7.2.1.6-cuda-11.0
module load cudnn/v8.0.5.39-prod-cuda-11.0

python3 main.py 