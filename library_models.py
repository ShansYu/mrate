'''
This is a supporting library with the code of the model.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv
import json

PATH = "./"

try:
    get_ipython
    trange = tnrange
    tqdm = tqdm_notebook
except NameError:
    pass

total_reinitialization_count = 0

# # MULTI-RELATION
# class RELATION():
#     def __init__(self, node, time, weight):
#         self.node = node
#         self.time = time
#         self.weight = weight

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE JODIE MODULE
class JODIE(nn.Module):
    def __init__(self, args, num_features, num_users, num_items, num_heads, dropout):
        super(JODIE,self).__init__()

        print("*** Initializing the JODIE model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.num_heads = num_heads
        self.dropout = dropout

        print("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features

        print("Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)
        self.his_multi_head_attention = nn.MultiheadAttention(self.embedding_dim, self.num_heads, self.dropout)
        self.com_multi_head_attention = nn.MultiheadAttention(self.embedding_dim, self.num_heads, self.dropout)
        self.seq_multi_head_attention = nn.MultiheadAttention(self.embedding_dim, self.num_heads, self.dropout)
        self.self_attention = nn.MultiheadAttention(3*self.embedding_dim, 1)

        print("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        # self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.prediction_layer = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        self.p_layer = nn.Linear(2, 1)
        self.attention_out_layer = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        print("*** JODIE initialization complete ***\n\n")
    
    def neighbor_aggregate(self, id_embedding, his_embeddings=None, com_embeddings=None, seq_embeddings=None):
        # his
        his_query = torch.LongTensor(id_embedding.repeat(len(his_embeddings), 1)).cuda()
        his_key = torch.LongTensor(his_embeddings).cuda()
        his_attn_output, his_attn_output_weights = self.his_multihead_attn(his_query, his_key, his_key)
        # com
        com_query = torch.LongTensor(id_embedding.repeat(len(com_embeddings), 1)).cuda()
        com_key = torch.LongTensor(com_embeddings).cuda()
        com_attn_output, com_attn_output_weights = self.com_multihead_attn(com_query, com_key, com_key)
        # seq
        seq_query = torch.LongTensor(id_embedding.repeat(len(seq_embeddings), 1)).cuda()
        seq_key = torch.LongTensor(seq_embeddings).cuda()
        seq_attn_output, seq_attn_output_weights = self.seq_multihead_attn(seq_query, seq_key, seq_key)
        # self_attention
        input_emb = torch.cat([his_attn_output, com_attn_output, seq_attn_output], dim=1)
        attn_output, attn_output_weights = self.self_attention(input_emb, input_emb, input_emb)
        output = self.attention_out_layer(torch.flatten(attn_output))
        return output

    def forward(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            #user_projected_embedding = torch.cat([input3, item_embeddings], dim=1)
            return user_projected_embedding
            
    def trans_p(self, item_embeddings, weight_t=None, weight_w=None):
        p = self.p_layer(torch.cat([weight_t, weight_w], dim=1))
        new_emebddings = p*item_embeddings
        return new_emebddings

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

def get_historical_neighbor(sorc_id, dest_id, t, his_dict):
    if sorc_id not in his_dict:
        his_dict[id] = [(dest_id, t, 1)]
    else:
        neighbor_node = [relation[0] for relation in his_dict[sorc_id]]
        if dest_id in neighbor_node:
            idx = neighbor_node.index(dest_id)
            his_dict[sorc_id][idx][1] = t
            his_dict[sorc_id][idx][2] += 1
        else:
            his_dict[sorc_id].append((dest_id, t, 1))
    return his_dict

def get_common_neighbor(id, t, T, com_dict, T_sequence):
    # T_sequence.append((id, t))
    for i in range(len(T_sequence)-1, -1, -1):
        if t - T_sequence[i][1] > T:
            T_sequence = T_sequence[i+1:]
            break
    if len(T_sequence) > 1:
        for node in T_sequence[:-1]:
            if node[0] not in com_dict:
                com_dict[node[0]] = [(id, t, 1)]
            else:
                neighbor_node = [relation[0] for relation in com_dict[node[0]]]
                if id in neighbor_node:
                    idx = com_dict[node[0]].index(id)
                    com_dict[node[0]][idx][1] = t
                    com_dict[node[0]][idx][2] += 1
                else:
                    com_dict[node[0]].append((id, t, 1))
            # 反向
            if id not in com_dict:
                com_dict[id] = [(node[0], t, 1)]
            else:
                neighbor_node = [relation[0] for relation in com_dict[id]]
                if node[0] in neighbor_node:
                    idx = com_dict[id].index(node[0])
                    com_dict[id][idx][1] = t
                    com_dict[id][idx][2] += 1
                else:
                    com_dict[id].append((node[0], t, 1))
    return com_dict

def get_seq_sim_neighbor(type, batch_idx, seq_dict):
    # user1:user2,t2,w2|user3,t3,w3
    f = open("seq_neighbor/"+ type + "/batch_" + str(batch_idx) + ".txt","r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(":")
        node_list = ls[1].strip().split("|")
        if ls[0] not in seq_dict:
            seq_dict[ls[0]] = []
        for each in node_list:
            node, time, weight = each.strip().split(",")
            seq_dict[ls[0]].append((node, time, weight))
    f.close()
    return seq_dict

def initialize_neighbors():
    global his_user2item, his_item2user, com_user2user, com_item2item, seq_user2user, seq_item2item
    his_user2item = defaultdict(list)
    his_item2user = defaultdict(list)
    com_user2user = defaultdict(list)
    com_item2item = defaultdict(list)
    seq_user2user = defaultdict(list)
    seq_item2item = defaultdict(list)
    # T_user_sequence = []
    # T_item_sequence = []

def get_max_neighbor_idx(id, realtion_dict, max_neighbor_idx, tbatch_start_time, tbatchid):
    for node_info in realtion_dict[id]:
        if tbatch_start_time is None or node_info[1] < tbatch_start_time:
            continue
        else:
            max_neighbor_idx = max(max_neighbor_idx, tbatchid[node_info[0]])
    return max_neighbor_idx

# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).cuda()[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s' % args.network)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.model, epoch, args.train_proportion))
    torch.save(state, filename)
    print("*** Saved embeddings and model to file: %s ***\n\n" % filename)


# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):
    modelname = args.model
    filename = PATH + "saved_models/%s/checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.network, modelname, epoch, args.train_proportion)
    checkpoint = torch.load(filename)
    print("Loading saved embeddings and model: %s" % filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()


# SELECT THE GPU WITH MOST FREE MEMORY TO SCHEDULE JOB 
def select_free_gpu():
    mem = []
    gpus = list(set(range(torch.cuda.device_count()))) # list(set(X)) is done to shuffle the array
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    return str(gpus[np.argmin(mem)])

