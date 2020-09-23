#!/bin/bash

echo Train
cd "$0"/src
time python sn7_data_prep.py $1
time python sn7_baseline_train.py $1
