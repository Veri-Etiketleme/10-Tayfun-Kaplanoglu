#! /bin/bash

#SBATCH -A SNIC2018-5-55
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --exclusive


ml icc/2017.1.132-GCC-5.4.0-2.26

ml ifort/2017.1.132-GCC-5.4.0-2.26
ml CUDA/8.0.44 impi/2017.1.132
ml Python/3.6.1
ml Tensorflow/1.3.0-Python-3.6.1
ml Keras/2.1.2-Python-3.6.1
ml matplotlib/2.0.1-Python-3.6.1 scikit-learn/0.18.1-Python-3.6.1
export KERAS_BACKEND=tensorflow

# keras was compield with mpi on kebnekaise, therefore we have to use "srun python" instead of "python"
srun python GEFS_error_regression_CNN_kebnekaise_v33.py
