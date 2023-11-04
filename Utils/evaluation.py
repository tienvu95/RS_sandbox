import time
import copy
import math
import numpy as np
import torch


from Utils.data_utils import to_np
			


def latent_factor_evaluate(model, test_dataset):
	"""
	Evaluation for latent factor model (BPR, LightGCN)
	----------

	Parameters
	----------
	model: model
	test_dataset: test dataset 

	Returns
	-------
	eval_results : dict
		summarizes the evaluation results
	"""
	
	metrics = {'H05':[], 'N05':[], 'H10':[], 'N10':[], 'H20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	# extract score 
	if model.type == 'MF':
		user_emb, item_emb = model.get_embedding()
	elif model.type == 'graph':
		user_emb, item_emb = model.computer()

	score_mat = to_np(-torch.matmul(user_emb, item_emb.T)) 
	test_user_list = to_np(test_dataset.user_list) 
	
	for test_user in test_user_list: 

		test_item = [int(test_dataset.test_item[test_user][0])] 
		valid_item = [int(test_dataset.valid_item[test_user][0])] 
		candidates = to_np(test_dataset.candidates[test_user]).tolist()

		total_items = test_item + valid_item + candidates
		score = score_mat[test_user][total_items] 
		
		result = np.argsort(score).flatten().tolist()
		ranking_list = np.array(total_items)[result]

		for mode in ['test', 'valid']:
			if mode == 'test':
				target_item = test_item[0] 
				ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == valid_item[0]))
			else:
				target_item = valid_item[0]
				ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == test_item[0]))
		
			for topk in ['05', '10', '20']:
				(h, n) = LOO_check(ranking_list_tmp, target_item, int(topk))
			
				eval_results[mode]['H' + topk].append(h)
				eval_results[mode]['N' + topk].append(n)

	# valid, test
	for mode in ['test', 'valid']:
		for topk in ['05', '10', '20']:
			eval_results[mode]['H' + topk] = round(np.asarray(eval_results[mode]['H' + topk]).mean(), 4)
			eval_results[mode]['N' + topk] = round(np.asarray(eval_results[mode]['N' + topk]).mean(), 4)	

	return eval_results

	
def evaluation(model, gpu, eval_dict, epoch, test_dataset):
	"""
	Parameters
	----------
	model: model
	gpu: gpu device
	eval_dict (dict): for control the training process
	epoch (int): current epoch
	test_dataset: test dataset

	Returns
	-------
	is_improved: is the result improved compared to the last best results
	eval_results: summary of the evaluation results
	toc-tic: elapsed time for evaluation
	"""

	model.eval()
	with torch.no_grad():
		tic = time.time()

		# NeuMF
		if model.type == 'network':
			eval_results = net_evaluate(model, gpu, test_dataset)

		# BPR, LightGCN
		elif model.type == 'MF' or model.type == 'graph':
			eval_results = latent_factor_evaluate(model, test_dataset)

		else:
			assert 'Unknown model type'	

		toc = time.time()
		is_improved = False

		for topk in ['05', '10', '20']:
			if eval_dict['early_stop'] < eval_dict['early_stop_max']:
				if eval_dict[topk]['best_score'] < eval_results['valid']['H' + topk]:
					eval_dict[topk]['best_score'] = eval_results['valid']['H' + topk]
					eval_dict[topk]['best_result'] = eval_results['valid']
					eval_dict[topk]['final_result'] = eval_results['test']

					is_improved = True
					eval_dict['final_epoch'] = epoch

		if not is_improved:
			eval_dict['early_stop'] +=1
		else:
			eval_dict['early_stop'] = 0

		return is_improved, eval_results, toc - tic
