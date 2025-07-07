import argparse

def parse_arguments(): 
    parser = argparse.ArgumentParser()

    ## seed and device
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)

    ## data
    parser.add_argument('--genome_embed_path', default='embed_matrix/ad_knuh_adult_dnabert_emb_mean_124_768_matrix.npy', type=str)
    parser.add_argument('--rand_path', default='embed_matrix/ad_knuh_adult_rand_emb_124_64_matrix.npy', type=str)
    parser.add_argument('--coat_path', default='graph/ad_knuh_adult_data_coat_thres_abs_0_1_random_sampling_coat_dnabert_64_abundance_uniform_64.pkl', type=str)

    ## model
    parser.add_argument('--input_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--output_dim', default=32, type=int)
    parser.add_argument('--clf_hidden_dim', default=32, type=int)
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--init_embed_dim', default=768, type=int)
    
    ## training     
    parser.add_argument('--lr', default=0.0001, type=float) 
    parser.add_argument('-e', '--epochs', default=70, type=int)
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--clf_drop_rate', default=0.3, type=float)
    parser.add_argument('--lambda_scale_abn', default=1.0, type=float)
    parser.add_argument('--lambda_scale_dnabert', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument("--save_checkpoint", action='store_true')
    parser.add_argument("--train_val_concat", action='store_true')
    
    return parser.parse_args()