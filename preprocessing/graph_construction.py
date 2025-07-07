import os
import argparse
import pickle
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data as GeometricData

class LoadData():
    def __init__(self, data_path, corr_path, coat_threshold = 0.1):
        self.data_path = data_path
        self.corr_path = corr_path
        self.coat_threshold = coat_threshold
    
    def load_data_corr(self):
        print('load data...')
        print(self.data_path)
        print(self.corr_path)
        
        with open(self.data_path, 'rb') as f:
            train_data = pickle.load(f)

        genus_coat = pd.read_csv(self.corr_path)
        genus_coat = genus_coat.rename(columns={'COAT': 'Corr'})

        ##### Edge info #####
        ## drop NA
        genus_coat = genus_coat[genus_coat['Corr'] != 'NA']
        ## coat threshold
        genus_coat['Corr'] = genus_coat['Corr'].astype(float)
        if isinstance(self.coat_threshold, list):
            print(self.coat_threshold)
            genus_coat_source = genus_coat[(genus_coat['Corr'] >= self.coat_threshold[0]) | (genus_coat['Corr'] <= self.coat_threshold[1])]
        elif self.coat_threshold == 999:
            genus_coat_source = genus_coat
        else:
            print(self.coat_threshold)
            genus_coat_source = genus_coat[genus_coat['Corr'] >= self.coat_threshold]
        genus_coat_source = genus_coat_source.reset_index(drop=True)
        
        return train_data, genus_coat_source

    def graph_construction_torch_geometric(self, data_info_x, data_info_y, edge_info, cols):
        graph_dataset = []
        genus_dict = {genus: idx for idx, genus in enumerate(cols)}

        for sample_x, sample_y in zip(data_info_x, data_info_y):
            non_zero_abundance_idxs = sample_x.nonzero()[0]
            
            non_zero_abundance = sample_x[non_zero_abundance_idxs]
            non_zero_cols = cols[non_zero_abundance_idxs]

            non_zero_genus_dict = {genus: idx for idx, genus in enumerate(non_zero_cols)}

            sample_edge = edge_info.loc[
                edge_info['variable_1'].isin(non_zero_genus_dict) &
                edge_info['variable_2'].isin(non_zero_genus_dict)
            ].copy()
            sample_edge['variable_1'] = sample_edge['variable_1'].map(non_zero_genus_dict)
            sample_edge['variable_2'] = sample_edge['variable_2'].map(non_zero_genus_dict)

            node_ids = np.vstack([
                np.concatenate([sample_edge['variable_1'], sample_edge['variable_2']]),
                np.concatenate([sample_edge['variable_2'], sample_edge['variable_1']])
            ])
            pos_corr_mask = np.where(sample_edge['Corr'] > 0.0)[0]
            pos_corr_mask_flip = np.where(sample_edge['Corr'] > 0.0)[0] + sample_edge['variable_1'].shape[0]
            neg_corr_mask = np.where(sample_edge['Corr'] < 0.0)[0]
            neg_corr_mask_flip = np.where(sample_edge['Corr'] < 0.0)[0] + sample_edge['variable_1'].shape[0]

            pos_corr_mask = np.concatenate([pos_corr_mask, pos_corr_mask_flip])
            neg_corr_mask = np.concatenate([neg_corr_mask, neg_corr_mask_flip])

            edge_mask = node_ids.copy()
            edge_mask[:, pos_corr_mask] = 1
            edge_mask[:, neg_corr_mask] = 0
            edge_mask = edge_mask[0]

            # Tensor
            abundance_fea = torch.tensor(non_zero_abundance, dtype=torch.float).unsqueeze(1)
            edge_index = torch.tensor(node_ids, dtype=torch.long)
            x_ids = torch.tensor(np.vstack([genus_dict[k] for k in non_zero_cols]), dtype=torch.int)
            non_zero_x_ids = torch.tensor([1 if i in non_zero_abundance_idxs else 0 for i in range(len(sample_x))], dtype=torch.int)

            data = GeometricData(
                x = abundance_fea,
                y = torch.tensor([int(sample_y)]),
                edge_index = edge_index,
                x_genus = non_zero_cols,
                x_ids = x_ids,
                non_zero_x_ids = non_zero_x_ids,
            )
            
            graph_dataset.append(data)

        return graph_dataset
    
    def graph_construction(self, graph_output_path, embed_path = None, random_fea_dim = 64 , random_fea_distr = 'uniform', store_embed_matrix = False):
        
        train_data, corr = self.load_data_corr()

        train_x = train_data['train_x']
        val_x = train_data['val_x']
        test_x = train_data['test_x']
        train_y = train_data['train_y']
        val_y = train_data['val_y']
        test_y = train_data['test_y']
        
        cols = np.array(train_data['microbes'])

        if embed_path is not None:
            with open(embed_path, 'rb') as f:
                seq_embed = pickle.load(f)
        else:
            print('Warning: Not include Embedding!!!!')

        if random_fea_distr == 'uniform':
            np.random.seed(42)
            random_fea = np.random.uniform(0, 1, size=(1, random_fea_dim)).astype(np.float32)
        elif random_fea_distr == 'gaussian':
            random_fea = np.random.randn(1, random_fea_dim).astype(np.float32)
        

        if store_embed_matrix:
            seq_embed_matrix = torch.tensor(np.vstack([seq_embed[k] for k in cols]), dtype=torch.float)
            seq_embed_matrix_path = '../datasets/processed/dnabert_emb_mean_124_768_matrix.npy'
            with open(seq_embed_matrix_path, 'wb') as f:
                np.save(f, seq_embed_matrix)
            print(seq_embed_matrix_path)

            rand_embed_matrix = torch.tensor(np.repeat(random_fea, 124, axis=0), dtype=torch.float)
            rand_embed_matrix_path = '../datasets/processed/rand_emb_124_64_matrix.npy'
            with open(rand_embed_matrix_path, 'wb') as f:
                np.save(f, rand_embed_matrix)
            print(rand_embed_matrix_path)
            print(f'genome embedding shape : {seq_embed_matrix.shape}')
            print(f'rand embedding shape : {rand_embed_matrix.shape}')
        

        train_dataset = self.graph_construction_torch_geometric(train_x, train_y, corr.copy(), cols)
        val_dataset = self.graph_construction_torch_geometric(val_x, val_y, corr.copy(), cols)
        test_dataset = self.graph_construction_torch_geometric(test_x, test_y, corr.copy(), cols)

        graph_dataset = {
            'train_data': train_dataset,
            'val_data': val_dataset,
            'test_data': test_dataset,
        }
        print(f'store path : {graph_output_path}')
        with open(graph_output_path, 'wb') as f:
            pickle.dump(graph_dataset, f)

        print('Train Dataset: {}'.format(len(train_dataset)))
        print('Val Dataset: {}'.format(len(val_dataset)))
        print('Test Dataset: {}'.format(len(test_dataset)))
        
        print('done.')
        breakpoint()
        return train_data, train_dataset, val_dataset, test_dataset, corr, cols, train_x, train_y
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name", type=str, default='ad_knuh_adult')
    parser.add_argument("--random_fea_dim", type=int, default=64)
    parser.add_argument("--random_fea_distr", type=str, default='uniform')
    parser.add_argument("--data_path", type=str, default='../datasets/processed/ad_knuh_adult_sum100.pkl')
    parser.add_argument("--coat_path", type=str, default='../datasets/coat/intersection_random_sampling_coat.csv')
    parser.add_argument("--embed_path", type=str, default='../datasets/embed_matrix/ad_knuh_public_adult_genus_dnabert_emb_mean.pkl')
    parser.add_argument("--store_embed_matrix", action='store_true')

    args, _ = parser.parse_known_args()

    coat_threshold = [0.1, -0.1] # COAT threshold

    loader = LoadData(
        data_path = args.data_path,
        corr_path = args.coat_path,
        coat_threshold = coat_threshold
    )

    if isinstance(coat_threshold, list):
        thres = f'abs_{coat_threshold[0]}'
        thres = thres.replace('.', '_')
    else:
        thres = f'{coat_threshold}'
        thres = thres.replace('.', '_')

    train_data, train_dataset, val_dataset, test_dataset, corr, cols, train_x, train_y = loader.graph_construction(
        random_fea_dim = args.random_fea_dim, # random vector dimension
        random_fea_distr = args.random_fea_distr, # random vector distribution
        ## DNABERT-2 embeddings
        embed_path = args.embed_path,
        ## COAT
        graph_output_path = f'../datasets/processed/{args.target_name}_data_coat_thres_{thres}_random_sampling_coat_dnabert_abundance_{args.random_fea_distr}_{args.random_fea_dim}.pkl',
        ## store dnabert and random embedding matrix
        store_embed_matrix = args.store_embed_matrix
    )