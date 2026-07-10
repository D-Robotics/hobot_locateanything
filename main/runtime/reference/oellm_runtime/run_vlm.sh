#!/bin/sh

export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6

config_file=$1
image_file=$2
if [ "$#" -ge 2 ]; then
    image_file=$2
    ./vlm -c $config_file -i $image_file
else
    ./vlm -c $config_file
fi
