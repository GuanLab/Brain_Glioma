#!/bin/bash

# Script for the whole training and prediction phases.

for i in 0 
do
    rm -rf result*
    perl split_gold_standard_by_record.pl ${i}
    python3 -u train_2d_simple_generator.py > record_0.txt
    cp weights.h5 weights_0.h5
    python3 -u train_2d_simple_generator.py > record_1.txt
    cp weights.h5 weights_1.h5
    python3 -u train_2d_simple_generator.py > record_2.txt
    cp weights.h5 weights_2.h5
    python3 -u train_2d_simple_generator.py > record_3.txt
    cp weights.h5 weights_3.h5
    python3 -u train_2d_simple_generator.py > record_4.txt
    cp weights.h5 weights_4.h5
    
    perl split_gold_standard_by_record_1staxis.pl ${i}
    python3 -u pred_1staxis.py > record_1st.txt
    mv result result_1staxis
    perl split_gold_standard_by_record_3rdaxis.pl ${i}
    python3 -u pred_3rdaxis.py > record_3rd.txt
    mv result result_3rdaxis
    perl split_gold_standard_by_record_2ndaxis.pl ${i}
    python3 -u pred_2ndaxis.py > record_2nd.txt
    mv result result_2ndaxis
    python by_head_nii_save_threeviews.py
    python evaluate_combine_dice_new.py
#    cp auc_auprc.txt auc_auprc_${i}.txt
#    cp eva_global.txt eva_global_${i}.txt

done
