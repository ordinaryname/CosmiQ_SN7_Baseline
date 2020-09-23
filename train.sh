#!/bin/bash

echo Train
cd $(dirname "$0")
time python sn7_data_prep.py $1 output
time python train.py output
