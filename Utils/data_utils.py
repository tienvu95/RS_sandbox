import pandas as pd


def to_np(x):
	return x.data.cpu().numpy()


def is_visited(base_dict, user_id, item_id):
    if user_id in base_dict and item_id in base_dict[user_id]:
        return True
    else:
        return False
    

def df_to_mat(df):
    result = {}
    for user_id, item_id in zip(df['user_id'], df['item_id']):
        if user_id not in result.keys():
            result[user_id] = {}
        result[user_id].update({item_id:1})
    sorted_keys = sorted(list(result.keys()))
    sorted_result = {i: result[i] for i in sorted_keys}
    
    return sorted_result


def dict_to_list(base_dict):
    result = []

    for user_id in base_dict:
        for item_id in base_dict[user_id]:
            result.append((user_id, item_id, 1))
    
    return result


def read_dataset(path, dataset):
    train_df = pd.read_pickle(f'{path}/{dataset}/train_df.pkl')
    train_mat = df_to_mat(train_df) 
    train_interactions = dict_to_list(train_mat)
    valid_df = pd.read_pickle(f'{path}/{dataset}/tune_df.pkl') 
    test_df = pd.read_pickle(f'{path}/{dataset}/test_df.pkl')

    return train_df, train_mat ,train_interactions, valid_df, test_df
