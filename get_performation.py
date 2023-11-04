import argparse
import torch
import torch.optim as optim

from Models.BPR import BPR
from Models.NeuMF import NeuMF
from Models.LightGCN import LightGCN
from Utils.data_utils import read_dataset

import gen_graph
from Utils.data_utils import to_np
from Utils.utility import test_model_all, print_results


def run():
    # gpu setting
    gpu = torch.device('cuda:' + str(opt.gpu))

    # dataset
    train_df, train_mat, train_interactions, valid_df, test_df = read_dataset(opt.data_path, opt.dataset)
    num_users = len(train_df['user_id'].unique())
    num_items = len(train_df['item_id'].unique())

    print(f'Model: {opt.model}, Dim: {opt.dim}, Dataset: {opt.dataset}, Num. users: {num_users}, Num. items: {num_items}')

    # setup model
    if opt.model == 'BPR':
        model = BPR(num_users, num_items, opt.dim, gpu)
    elif opt.model == 'NeuMF':
        model = NeuMF(num_users, num_items, opt.dim, opt.num_layers, gpu)
    elif opt.model == 'LightGCN':
        Graph = gen_graph.getSparseGraph(train_mat, num_users, num_items, gpu)
        model = LightGCN(num_users, num_items, opt.dim, opt.num_layers, Graph, gpu)
    else:
        assert False


    with torch.no_grad():
        weight_path = f'{opt.model_path}/{opt.model}_{opt.dataset}'
        model = model.to(gpu)
        model.load_state_dict(torch.load(weight_path, map_location='cuda:' + str(opt.gpu)))
        if model.type == 'MF' or model.type == 'graph':
            user_emb, item_emb = model.get_embedding()
            Rec = to_np(torch.matmul(user_emb, item_emb.T))
        elif model.type == 'network':
            Rec = []
            batch_user = model.user_list
            batch_items = [torch.LongTensor([i]*batch_user.shape[0]).to(opt.gpu) for i in range(model.item_count)]
            for batch_item in batch_items:
                score = model.forward_no_neg(batch_user, batch_item)
                score = score.squeeze(-1).tolist()
                Rec.append(score)
            Rec = np.array(Rec).T
        precision, recall, f_score, NDCG = test_model_all(Rec, test_df, train_df)            
        print_results(precision, recall, f_score, NDCG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_path', type=str, default='Data sets/')
    parser.add_argument('--dataset', type=str, default='amazon_2')

    # model
    parser.add_argument('--model_path', type=str, default='Saved models/')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--model', type=str, default='BPR')
    parser.add_argument('--dim', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers (for NeuMF and LightGCN)')

    opt = parser.parse_args()
    # print(opt)

    run()
