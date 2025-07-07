import os
import argparse
import pandas as pd
import numpy as np
import pickle

from abun_utils import process_genus_df, remove_zero_sum_rows_cols, train_test_val_split

def load_data(data_path = None):
    if data_path is None:
        ad_korea_univ = pd.read_csv('../datasets/abundance/ad_korea.csv').drop(columns='index')
        ad_hka = pd.read_csv('../datasets/abundance/ad_hka.csv').drop(columns='index')
        ad_cn = pd.read_csv('../datasets/abundance/scn_atopy.csv', delimiter=',')
        ad_cn_normal = pd.read_csv('../datasets/abundance/ad_cn.csv')
        ad_jp_normal = pd.read_csv('../datasets/abundance/cntr_jp.csv')
        ad_knuh = pd.read_csv('../datasets/abundance/knuh_samples.csv')

        ad_cn['Label'] = ['atopy' if i == 'AD' else 'normal' for i in ad_cn['Label']]

        ## extract genus
        ad_korea_univ_genus = process_genus_df(ad_korea_univ.copy().iloc[:,:-1])
        ad_korea_univ_genus = pd.concat([ad_korea_univ_genus, ad_korea_univ['Label']], axis=1)

        ad_hka_genus = process_genus_df(ad_hka.copy().iloc[:,:-1])
        ad_hka_genus = pd.concat([ad_hka_genus, ad_hka['Label']], axis=1)

        ad_cn_genus = process_genus_df(ad_cn.copy().iloc[:,:-1])
        ad_cn_genus = pd.concat([ad_cn_genus, ad_cn['Label']], axis=1)

        ad_cn_normal_genus = process_genus_df(ad_cn_normal.copy().iloc[:,:-1])
        ad_cn_normal_genus = pd.concat([ad_cn_normal_genus, ad_cn_normal['Label']], axis=1)

        ad_jp_normal_genus = process_genus_df(ad_jp_normal.copy().iloc[:,:-1])
        ad_jp_normal_genus = pd.concat([ad_jp_normal_genus, ad_jp_normal['Label']], axis=1)

        ad_knuh_genus = process_genus_df(ad_knuh.copy().iloc[:,:-1])
        ad_knuh_genus = pd.concat([ad_knuh_genus, ad_knuh['Label']], axis=1)

        ## knuh data
        ad_knuh_adult = ad_knuh_genus.copy()

        ## public data
        ad_public_adult = pd.concat([ad_korea_univ_genus, ad_hka_genus, ad_cn_genus, ad_cn_normal_genus, ad_jp_normal_genus], axis=0).fillna(0).reset_index(drop=True)
        ad_public_adult = pd.concat([ad_public_adult.drop(columns='uncultured'), ad_public_adult['uncultured']], axis=1)
        ad_public_adult = pd.concat([ad_public_adult.drop(columns='no_genus'), ad_public_adult['no_genus']], axis=1)
        ad_public_adult = pd.concat([ad_public_adult.drop(columns='Label'), ad_public_adult['Label']], axis=1)

        ## drop no_genus, uncultured
        ad_public_adult = ad_public_adult.drop(columns=['no_genus', 'uncultured']) 
        ad_knuh_adult = ad_knuh_adult.drop(columns=['no_genus', 'uncultured'])
        ## drop row, column (sum 0)
        ad_public_adult = remove_zero_sum_rows_cols(ad_public_adult)
        ad_knuh_adult = remove_zero_sum_rows_cols(ad_knuh_adult)

        return ad_public_adult, ad_knuh_adult
    else:
        ad_df = pd.read_csv(data_path)
        ## extract genus
        ad_df_genus = process_genus_df(ad_df.copy().iloc[:,:-1])
        ad_df_genus = pd.concat([ad_df_genus, ad_df['Label']], axis=1)
        ## drop no_genus, uncultured
        ad_df_genus = ad_df_genus.drop(columns=['no_genus', 'uncultured']) 
        ## drop row, column (sum 0)
        ad_df_genus = remove_zero_sum_rows_cols(ad_df_genus)

        return ad_df_genus

def store_whole_data(ad_public_adult, ad_knuh_adult, custom_data = False):
    if custom_data:
        return None
    
    ad_public_knuh_adult = pd.concat([ad_public_adult, ad_knuh_adult], axis=0).fillna(0).reset_index(drop=True)
    ad_public_knuh_adult = pd.concat([ad_public_knuh_adult.drop(columns='Label'), ad_public_knuh_adult['Label']], axis=1)

    os.makedirs('../datasets/processed', exist_ok=True)
    ad_public_knuh_adult.to_csv(f'../datasets/processed/ad_public_knuh_adult.csv', index = False)

def split_data(ad_df, n_sum = 100, custom_data = False):

    data_name = 'ad_knuh_adult'
    if custom_data:
        data_name = 'custom_data'

    df_per_row_sum = ad_df.iloc[:,:-1].sum(axis=1)
    ad_df.iloc[:,:-1] = ad_df.iloc[:,:-1].div(df_per_row_sum, axis=0) * n_sum

    ## split
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    ad_df_train, ad_df_val, ad_df_test = train_test_val_split(ad_df, train_ratio = train_ratio, validation_ratio = val_ratio, test_ratio = test_ratio, seed = 42)

    ## store csv
    os.makedirs('../datasets/processed', exist_ok=True)
    ad_df_train.to_csv(f'../datasets/processed/{data_name}_sum{n_sum}_train.csv', index = False)
    ad_df_val.to_csv(f'../datasets/processed/{data_name}_sum{n_sum}_val.csv', index = False)
    ad_df_test.to_csv(f'../datasets/processed/{data_name}_sum{n_sum}_test.csv', index = False)

    ## store pkl
    ad_df_data = {
        'train_x':ad_df_train.iloc[:,:-1].values,
        'val_x':ad_df_val.iloc[:,:-1].values,
        'test_x':ad_df_test.iloc[:,:-1].values,
        'train_y':np.array([0 if i == 'normal' else 1 for i in ad_df_train['Label']]),
        'val_y':np.array([0 if i == 'normal' else 1 for i in ad_df_val['Label']]),
        'test_y':np.array([0 if i == 'normal' else 1 for i in ad_df_test['Label']]),
        'microbes':ad_df_train.iloc[:,:-1].columns.tolist(),
    }
    with open(f'../datasets/processed/{data_name}_sum{n_sum}.pkl', 'wb') as f:
        pickle.dump(ad_df_data, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--n_sum", type=int, default=100)

    args, _ = parser.parse_known_args()

    if args.data_path is None:
        ad_public_adult, ad_knuh_adult = load_data()
        store_whole_data(ad_public_adult, ad_knuh_adult)
        split_data(ad_knuh_adult)
    else:
        ad_df = load_data(data_path = args.data_path)
        split_data(ad_df, n_sum = args.n_sum, custom_data = True)