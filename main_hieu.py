import argparse
import torch
import torch.utils.data as data
import torch.optim as optim

from Models.BPR import BPR
from Models.NeuMF import NeuMF
from Models.LightGCN import LightGCN
from Utils.dataset import implicit_CF_dataset
from Utils.data_utils import read_dataset

import gen_graph
from run import model_run    

def run():
    # gpu setting
	gpu = torch.device('cuda:' + str(opt.gpu))

	# dataset
	train_df, train_mat, train_interactions, valid_df, test_df = read_dataset(opt.data_path, opt.dataset)
	num_users = len(train_df['user_id'].unique())
	num_items = len(train_df['item_id'].unique())

	print(f'Model: {opt.model}, Dim: {opt.dim}, Dataset: {opt.dataset}, Num. users: {num_users}, Num. items: {num_items}')
	train_dataset = implicit_CF_dataset(num_users, num_items, train_mat, train_interactions, opt.num_ns)
	train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
 
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
	model = model.to(gpu)

	# training
	optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.reg)
	model_path = f'{opt.model_path}/{opt.model}_{opt.dataset}'
	model_run(opt, model, optimizer, train_loader, train_df, test_df, gpu, model_save_path=model_path)
	# model_run(opt, model, optimizer, train_loader, train_df, test_df, gpu, model_save_path=None)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# training
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--reg', type=float, default=0.001, help='weight decay')
	parser.add_argument('--batch_size', type=int, default=1024)
	parser.add_argument('--num_ns', type=int, default=1, help='number of negative samples')

	parser.add_argument('--n_epochs', type=int, default=500)
	parser.add_argument('--early_stop', type=int, default=0, help='number of epochs for early stopping')
	parser.add_argument('--model_path', type=str, default='Saved models/')
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')

	# dataset
	parser.add_argument('--data_path', type=str, default='Data sets/')
	parser.add_argument('--dataset', type=str, default='amazon_2')
	
	# model
	parser.add_argument('--model', type=str, default='BPR')
	parser.add_argument('--dim', type=int, default=20)
	parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers (for NeuMF and LightGCN)')

	opt = parser.parse_args()
	# print(opt)

	run()
