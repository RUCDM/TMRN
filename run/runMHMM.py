# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys, time
sys.path.append('..')

import numpy as np, time
import pandas as pd
import MHopMN4rec3_hop3 as MN4rec
#import MHopMN4rec_noCate2_hopk as MN4rec
import evaluation

PATH_TO_TRAIN = sys.argv[1] # '../data/movie_sparse/train.oneSessout.csv'

File_ItemEmbedding = sys.argv[2]
File_CateEmbedding = sys.argv[3]
File_TreePath = sys.argv[4]
def read_ItemEmbedding(File_ItemEmbedding):
    ItemEmbedding = {}
    f = open(File_ItemEmbedding)
    #length = int(f.readline().strip().split()[1])
    a = []
    for line in f.readlines():
        ss = line.strip().split()
	#if ss[0][0] != 'm':
	#    continue
        ItemId = int(ss[0][:])
        t = []
        for i in range(1, len(ss)):
            t.append(float(ss[i]))
        ItemEmbedding[ItemId] = np.array(t, dtype = np.float32)
	a.append(len(ss)-1)
    print set(a)
    f.close()
    length = ItemEmbedding[ItemEmbedding.keys()[0]].shape[0]
    return ItemEmbedding, length

if __name__ == '__main__':
    print 'python', " ".join(sys.argv)
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})

    ItemEmbedding, length = read_ItemEmbedding(File_ItemEmbedding)
    print "the length and number of data ItemEmbedding", length, len(ItemEmbedding)
    CateEmbedding, Clength = read_ItemEmbedding(File_CateEmbedding)
    print "the length and number of data ItemEmbedding", Clength, len(CateEmbedding)
    Paths = pd.read_csv(File_TreePath, sep='#', dtype=np.int32)
    #Reproducing results from "Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    '''    
    print('Training GRU4Rec with 100 hidden units')    
    
    gru = gru4rec.GRU4Rec(loss='top1', final_act='tanh', hidden_act='tanh', layers=[100], batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, time_sort=False)
    gru.fit(data)
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    
    '''
    #Reproducing results from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847) 
    print('Training GRU4Rec with 100 hidden units')
    start_time = time.time()
    #gru = MN4rec.GRU4Rec(loss='bpr', final_act='linear', hidden_act='tanh', layers=[256], batch_size=200, embedding=length, Cate_embedding=Clength, dropout_p_hidden=0.2, n_sample=10, learning_rate=0.02, momentum=0.1, sample_alpha=0, time_sort=True, n_epochs=40, train_random_order=True, out_dim=int(sys.argv[-1]))
    gru = MN4rec.GRU4Rec(loss='bpr', final_act='linear', hidden_act='tanh', layers=[256], batch_size=200, embedding=length, Cate_embedding=Clength, dropout_p_hidden=0.2, n_sample=10, learning_rate=0.001, momentum=0.1, sample_alpha=0, time_sort=True, n_epochs=20, train_random_order=True, out_dim=int(sys.argv[-1]))

    gru.fit(data, ItemEmbedding, CateEmbedding, Paths)
    print("Training time is")
    print (start_time - time.time())
    prefix = str(time.time())
    ItemFile = 'item_embedding.'+prefix
    #if len(sys.argv)>5:
    #	ItemFile = sys.argv[5]
    gru.save_ItemEmbedding(data, ItemFile)#'item.embedding')

    UserFile = 'user.embedding.' + prefix
    #if len(sys.argv)>6:
    #	UserFile = sys.argv[6]
    evaluation.evaluate_sessions_batch(gru, valid, None, SaveUserFile = UserFile)#'user.embedding')
    end_time = time.time()
    print start_time, end_time
    print (start_time - end_time)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))

