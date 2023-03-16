#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

# Please comment and uncomment the corresponding part to train and evaluate on 
# different datasets. [CASME | SAMM]

# for CASME

# SUB_LIST=( casme_015 casme_016 casme_019 casme_020 casme_021 casme_022 casme_023 casme_024 \
# casme_025 casme_026 casme_027 casme_029 casme_030 casme_031 casme_032 casme_033 casme_034 \
# casme_035 casme_036 casme_037 casme_038 casme_040 )
# OUTPUT="./output/casme"
# DATASET="cas(me)^2"

# for SAMM

SUB_LIST=( samm_007 samm_006 samm_008 samm_009 samm_010 samm_011 samm_012 samm_013 samm_014 \
samm_015 samm_016 samm_017 samm_018 samm_019 samm_020 samm_021 samm_022 samm_023 samm_024 \
samm_025 samm_026 samm_028 samm_030 samm_031 samm_032 samm_033 samm_034 samm_036 samm_035 \
samm_037 )
OUTPUT="./output/samm"
DATASET="samm"

for i in ${SUB_LIST[@]}
do     
    echo "************ Currently running subject: ${i} ************"
    # comment the line below if evaluating on available ckpts.
    #python train.py --dataset $DATASET --output $OUTPUT --subject ${i}  # for training
    python eval.py --dataset $DATASET --output $OUTPUT --subject ${i}   # for evaluation
done

#output final metrics
python calc_final_score.py --output $OUTPUT