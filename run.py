import time
import torch
import numpy as np
from Utils.data_utils import to_np
from Utils.utility import test_model_all, print_results


def model_run(opt, model, optimizer, train_loader, train_df, test_df, gpu, model_save_path=None):
    n_epochs = opt.n_epochs
    precision_best, recall_best, f_score_best, NDCG_best = \
            np.array([-1, -1, -1, -1]), np.array([-1, -1, -1, -1]), np.array([-1, -1, -1, -1]), np.array([-1, -1, -1, -1])
    # begin training
    
    for epoch in range(n_epochs):
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []

        # mini-batch training
        for batch_user, batch_pos_item, batch_neg_item in train_loader:
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)

            # forward propagation
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)

            # batch loss
            batch_loss = model.get_loss(output)
            epoch_loss.append(batch_loss)

            # update model parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # total loss in an epoch
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()

        # evaluation
        tic2 = time.time()
        print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Training time: {toc1-tic1:.4f}s')

        if opt.early_stop > 0:
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
            condition = (np.array(precision) > precision_best).sum() + \
                        (np.array(recall) > recall_best).sum() + \
                        (np.array(f_score) > f_score_best).sum() + \
                        (np.array(NDCG) > f_score_best).sum()
            
            if bool(condition > 1):
                is_improved = True
                not_improved_count = 0
            else:
                is_improved = False
                not_improved_count += 1    

            print(f'is_improved: {is_improved}, not_improved_count: {not_improved_count}')
            print_results(precision, recall, f_score, NDCG)

            # save model
            if model_save_path != None and is_improved:
                torch.save(model.state_dict(), model_save_path)
            
            if not_improved_count > opt.early_stop:
                break

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
    # save model
    if model_save_path != None:
        torch.save(model.state_dict(), model_save_path)
