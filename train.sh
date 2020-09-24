#!/bin/bash

echo Train
cd $(dirname "$0")
source activate solaris
time python src/sn7_data_prep.py $1 output/train
time python train.py
