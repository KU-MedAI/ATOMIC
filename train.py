import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from models.atomic import ATOMIC

import utils

    
def saveModel(model, file_name, epoch_step):
    path = f'./checkpoints/{file_name}_epoch{epoch_step}.pth'
    torch.save(model.state_dict(), path)
    print(f'[]checkpoint saved - {path}')

def eval_crossentropy(args, model, eval_loader, criterion):
    eval_loss = 0.0
    eval_auprc = 0.0
    eval_auroc = 0.0
    eval_f1 = 0.0

    pred_epoch = []
    true_epoch = []
    
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(eval_loader):
            eval_data = data.cuda()

            outputs,_,_,_,_ = model(eval_data.x, eval_data.edge_index, eval_data.batch, eval_data.x_ids)
            
            loss = criterion(outputs.squeeze(), eval_data.y.float())

            eval_loss += loss.item() # Track the loss value
            pred_cpu = outputs.squeeze().detach().cpu().numpy()
            y_cpu = eval_data.y.float().detach().cpu().numpy()
            pred_epoch.append(pred_cpu)
            true_epoch.append(y_cpu)

        true_values = np.concatenate((true_epoch),axis=0).squeeze().tolist()
        pred_values = np.concatenate((pred_epoch),axis=0).squeeze().tolist()
        pred_values_threshold = (np.concatenate((pred_epoch),axis=0).squeeze() >= 0.5).astype(int).tolist()
        metrics = [
            average_precision_score(true_values, pred_values),
            roc_auc_score(true_values, pred_values),
            f1_score(true_values, pred_values_threshold)
            ]

        eval_auprc = metrics[0]
        eval_auroc = metrics[1]
        eval_f1 = metrics[2]
        eval_loss /= len(eval_loader)

    return eval_loss, eval_auprc, eval_auroc, eval_f1, true_values, pred_values, pred_values_threshold

def train_crossentropy(args, model, epochs, train_loader, val_loader, test_loader, optimizer, criterion, scheduler):
    
    lowest_loss = float("inf")
    test_auroc = 0.0
    test_f1 = 0.0
    
    patience = 0
    patience_limit = 20
    least_epoch = 20
    status_least_epoch = False
    stop_epoch = 0

    for epoch in range(epochs):
        train_loss = 0.0

        pred_epoch = []
        true_epoch = []
        
        model.train()
        
        for idx, data in enumerate(train_loader):
            data = data.cuda()
            optimizer.zero_grad()

            outputs,_,_,_,_ = model(data.x, data.edge_index, data.batch, data.x_ids)
            
            loss = criterion(outputs.squeeze(), data.y.float())
            loss.backward() # Derive gradients
            optimizer.step() # Update parameters based on gradients

            train_loss += loss.item() # Track the loss value
            
            pred_cpu = outputs.squeeze().detach().cpu().numpy()
            y_cpu = data.y.float().detach().cpu().numpy()
            pred_epoch.append(pred_cpu)
            true_epoch.append(y_cpu)
            
        true_values = np.concatenate((true_epoch),axis=0).squeeze().tolist()
        pred_values = np.concatenate((pred_epoch),axis=0).squeeze().tolist()
        pred_values_threshold = (np.concatenate((pred_epoch),axis=0).squeeze() >= 0.5).astype(int).tolist()

        metrics = [round(average_precision_score(true_values, pred_values), 3),
                round(roc_auc_score(true_values, pred_values), 3),
                round(f1_score(true_values, pred_values_threshold), 3)]

        train_auprc = metrics[0]
        train_auroc = metrics[1]
        train_f1 = metrics[2]
        train_loss /= len(train_loader)
        
        val_loss, val_auprc, val_auroc, val_f1,_,_,_ = eval_crossentropy(args, model, val_loader, criterion)
        
        scheduler.step()
            
        print('Epoch {:02d}/{:02d} Train_Loss{:9.3f}, Valid_Loss{:9.3f}, | Train_AUROC {:9.3f}, Train_F1 {:9.3f}, | Valid_AUROC {:9.3f}, Valid_F1 {:9.3f}'.format(
                epoch+1, epochs, train_loss, val_loss, train_auroc, train_f1, val_auroc, val_f1))
        
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            stop_epoch = epoch + 1
            patience = 0
            ## minimum epoch
            if status_least_epoch:
                patience_limit = 20

            _, test_auprc, test_auroc, test_f1,_,_,_ = eval_crossentropy(args, model, test_loader, criterion)
            print('[TEST]Epoch {:02d}/{:02d} \t\t\t\t Test_AUPRC {:9.3f}, Test_AUROC {:9.3f}, Test_F1 {:9.3f}\n'.format(
                    epoch+1, epochs, test_auprc, test_auroc, test_f1))

        else:
            patience += 1
            if patience >= patience_limit:
                if args.train_val_concat:
                    pass
                else:
                    ## minimum epoch
                    if stop_epoch <= least_epoch and status_least_epoch == False:
                        status_least_epoch = True
                        patience = 0
                        patience_limit = 100
                    else:
                        break

    _, test_auprc, test_auroc, test_f1,_,_,_ = eval_crossentropy(args, model, test_loader, criterion)
    print('[TEST]Epoch {:02d}/{:02d} \t\t\t\t Test_AUPRC {:9.3f}, Test_AUROC {:9.3f}, Test_F1 {:9.3f}\n'.format(
            epoch+1, epochs, test_auprc, test_auroc, test_f1))

    # save_checkpoint
    if args.save_checkpoint:
        save_name = f'model_weight'
        saveModel(model, save_name, args.epochs)

if __name__ == '__main__':

    args = utils.parse_arguments()
    
    ### GPU 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    device = torch.device(f"cuda:{str(args.gpu_id)}" if torch.cuda.is_available() else "cpu")
    print(f'Count of using GPUs: {torch.cuda.get_device_name(args.gpu_id)} {torch.cuda.device_count()}')
    print(f'Current cuda device: {torch.cuda.get_device_name(args.gpu_id)} {device}')

    ## DNABERT-2 seq_embed matrix 
    seq_embed_matrix_path = f'./datasets/{args.genome_embed_path}'
    seq_embed_matrix = np.load(seq_embed_matrix_path)
    seq_embed_matrix = torch.FloatTensor(seq_embed_matrix)
    print(f'genome embedding shape : {seq_embed_matrix.shape}')
    ## random embed matrix
    rand_embed_matrix_path = f'./datasets/{args.rand_path}'
    rand_embed_matrix = np.load(rand_embed_matrix_path)
    rand_embed_matrix = torch.FloatTensor(rand_embed_matrix)
    print(f'rand init shape : {rand_embed_matrix.shape}')

    ### COAT
    file_path = f'./datasets/{args.coat_path}'
    
    with open(file_path, 'rb') as f:
        graph_data = pickle.load(f)
    train_dataset, val_dataset, test_dataset = graph_data['train_data'], graph_data['val_data'], graph_data['test_data']
    
    if args.train_val_concat:
        train_dataset = train_dataset + val_dataset

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    
    model = ATOMIC(input_dim = args.input_dim,
                   hidden_dim = args.hidden_dim,
                   output_dim = args.output_dim,
                   drop_rate = args.drop_rate,
                   num_layers = args.n_layers,
                   num_heads = args.n_heads,
                   
                   clf_hidden_dim = args.clf_hidden_dim,
                   clf_drop_rate = args.clf_drop_rate,
                   preinitialized_dnabert_embeddings = seq_embed_matrix,
                   preinitialized_random_fea_embeddings = rand_embed_matrix,
                   lambda_scale_abn = args.lambda_scale_abn,
                   lambda_scale_dnabert = args.lambda_scale_dnabert,
                   init_embed_dim = args.init_embed_dim,
                   ).to(device)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = 10,
        gamma = 0.99
    )

    try:
        train_crossentropy(args, model, args.epochs, train_loader, val_loader, test_loader, optimizer, criterion, scheduler)
    except KeyboardInterrupt:
        print('Terminating...')