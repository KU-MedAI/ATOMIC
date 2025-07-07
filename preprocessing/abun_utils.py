import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def process_genus_df(df, delimiter = ';'):
    cols_dict = {}
    cols = df.columns.tolist()
    temp_shape = df.shape

    for f in cols:
        splited = f.split(delimiter)
        name = next((part[3:] for part in splited if part.startswith('g__')), None)
        if name != None:
            if "uncultured" in name:
                name = 'uncultured'
            if "unclassified" in name:
                name = 'unclassified'
            if name != "":
                cols_dict[f] = name
        else:
            cols_dict[f] = 'no_genus'

    df = df.rename(cols_dict, axis='columns')
    df_uncultured = df.pop('uncultured').max(axis=1)
    df_uncultured.name = 'uncultured'
    df_na = df.pop('no_genus').max(axis=1)
    df_na.name = 'no_genus'
    df = df.astype(float)

    df = pd.concat([df, df_uncultured, df_na], axis=1)

    print(f'{temp_shape} -> {df.shape}')

    return df

def remove_zero_sum_rows_cols(df):

    numeric_df = df.iloc[:, :-1]
    row_sums = numeric_df.sum(axis=1)
    col_sums = numeric_df.sum(axis=0)

    rows_to_drop = row_sums[row_sums == 0.0].index
    cols_to_drop = col_sums[col_sums == 0.0].index

    df = df.drop(rows_to_drop)
    df = df.drop(cols_to_drop, axis=1)
    df = df.reset_index(drop=True)

    return df

def train_test_val_split(df, train_ratio = 0.6, validation_ratio = 0.2, test_ratio = 0.2, seed = 42):
    
    X = df.iloc[:,:-1]
    Y = df['Label'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size= 1 - train_ratio,
                                                        random_state=seed,
                                                        shuffle=True,
                                                        stratify=Y)

    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test,
                                                    test_size= test_ratio / (test_ratio + validation_ratio),
                                                    random_state=seed,
                                                    shuffle=True,
                                                    stratify=Y_test)


    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    X_train['Label'] = Y_train
    X_val['Label'] = Y_val
    X_test['Label'] = Y_test

    print(f'X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}')

    return X_train, X_val, X_test