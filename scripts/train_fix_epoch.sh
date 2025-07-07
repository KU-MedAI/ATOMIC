#!/bin/bash

python train.py \
--epochs 70 \
--train_val_concat \
--genome_embed_path embed_matrix/ad_knuh_adult_dnabert_emb_mean_124_768_matrix.npy \
--rand_path embed_matrix/ad_knuh_adult_rand_emb_124_64_matrix.npy \
--coat_path graph/ad_knuh_adult_data_coat_thres_abs_0_1_random_sampling_coat_dnabert_64_abundance_uniform_64.pkl \
--gpu_id 0