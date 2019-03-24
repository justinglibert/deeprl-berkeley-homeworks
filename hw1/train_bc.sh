#!/bin/bash
set -eux
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    for h in "100" "100,100" "100,100,100"
    do
        python behavioral_cloning.py --envname $e --hidden $h
    done
done
