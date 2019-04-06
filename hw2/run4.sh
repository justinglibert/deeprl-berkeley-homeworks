#!/bin/bash
rm -rf data/sb_* data/lb_*
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na
python.app plot.py data/sb_no_rtg_dna_CartPole-v0 data/sb_rtg_dna_CartPole-v0 data/sb_rtg_na_CartPole-v0 --value AverageReturn --legend sb_no_rtg_dna sb_rtg_dna sb_rtg_na --save_name p4_sb 
python.app plot.py data/lb_no_rtg_dna_CartPole-v0 data/lb_rtg_dna_CartPole-v0 data/lb_rtg_na_CartPole-v0 --value AverageReturn --legend lb_no_rtg_dna lb_rtg_dna lb_rtg_na --save_name p4_lb

