#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

#TODO: Create MODELPATH variable
MODELPATH=$1

#TODO: Call the Python script
python3 inference_cpu_model.py  --model_path ${MODELPATH}

cd /output

tar zcvf output.tgz stdout.log stderr.log