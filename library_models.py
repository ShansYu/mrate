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
# import cPickle
import pickle
import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv

PATH = "./"

try:
    get_ipython
    trange = tnrange
    tqdm = tqdm_notebook
except NameError:
    pass

total_reinitialization_count = 0

# his_user2item = defaultdict(list)
# his_item2user = defaultdict(list)
# com_user2user = defaultdict(list)
# com_item2item = defaultdict(list)
# seq_user2user = defaultdict(list)
# seq_item2item = defaultdict(list)

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE JODIE MODULE
class JODIE(nn.Module):
    def __init__(self, args, num_features, num_users, num_items):
        super(JODIE,self).__init__()

        print ("*** Initializing the JODIE model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.num_neighbor = args.num_neighbor
        self.num_self_attention_head = 50

        print ("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_items = rnn_input_size_users = 2*self.embedding_dim + 1

        print ("Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)

        print ("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        # multi-head attention
        self.attention_hidden_1  = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attention_hidden_2  = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attention_hidden_3  = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attention_weight_his_1 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_weight_his_2 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_weight_his_3 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_weight_com_1 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_weight_com_2 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_weight_com_3 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_weight_seq_1 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_weight_seq_2 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_weight_seq_3 = nn.Linear(2*self.embedding_dim, 1)
        self.attention_p = nn.Linear(2, 1)
        # self-attention
        self.query = nn.Linear(self.embedding_dim, self.num_self_attention_head)
        self.key = nn.Linear(self.embedding_dim, self.num_self_attention_head)
        self.value = nn.Linear(self.embedding_dim, self.num_self_attention_head)
        self.fc = nn.Linear(3*self.num_self_attention_head, self.embedding_dim)
        print ("*** JODIE initialization complete ***\n\n")

    def forward(self, user_embeddings, item_embeddings, user_neighbor_embeddings, item_neighbor_embeddings, timediffs=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, item_neighbor_embeddings, timediffs], dim=1)
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, user_neighbor_embeddings, timediffs], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)

        # elif select == 'project':
        #     user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
        #     #user_projected_embedding = torch.cat([input3, item_embeddings], dim=1)
        #     return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings, user_neighbor_embeddings):
        X_out = nn.Softmax()(self.prediction_layer(torch.cat([user_embeddings, user_neighbor_embeddings], dim=1)))
        return X_out

    def multi_head_attention(self, neighbor_embeddings, target_embeddings, t_tensor, w_tensor, hidden_layer, weight_layer):
        neighbor_embeddings = neighbor_embeddings.cpu()
        t_tensor = t_tensor.cpu()
        w_tensor = w_tensor.cpu()
        hidden_neighbor = hidden_layer(neighbor_embeddings)
        hidden_target = hidden_layer(target_embeddings)
        p = nn.Sigmoid()(self.attention_p(torch.cat([t_tensor, w_tensor], dim=2)))
        hidden_target = hidden_target.unsqueeze(1)
        # print('hidden_target shape:{}'.format(hidden_target.shape))
        # tmp = target_embeddings.repeat(1,self.num_neighbor,1)
        # print('target_embeddings shape:{}'.format(target_embeddings.shape))
        # print('target_embeddings.repeat shape: {}, neighbor_embeddings: {}'.format(tmp.shape, neighbor_embeddings.shape))
        # c = nn.LeakyReLU()(weight_layer(torch.cat([target_embeddings.repeat(1,self.num_neighbor,1), neighbor_embeddings], dim=2) * p))
        c = nn.LeakyReLU()(weight_layer(torch.cat([hidden_target.repeat(1,self.num_neighbor,1), neighbor_embeddings], dim=2) * p))
        a = nn.Softmax()(torch.reshape(c, (-1, self.num_neighbor)))
        a_reshape = torch.reshape(a, (-1, self.num_neighbor, 1))
        out = torch.reshape(torch.sum(hidden_neighbor * a_reshape, dim=1), (-1, self.embedding_dim, 1)) # bacth_size, dim, 1
        return out

    # def multi_head_attention(self, neighbor_embeddings, target_embeddings, t_tensor, w_tensor, hidden_layer, weight_layer):
    #     neighbor_embeddings = neighbor_embeddings.cpu()
    #     t_tensor = t_tensor.cpu()
    #     w_tensor = w_tensor.cpu()
    #     hidden_neighbor = hidden_layer(neighbor_embeddings)
    #     hidden_target = hidden_layer(target_embeddings)
    #     p = nn.Sigmoid()(self.attention_p(torch.cat([t_tensor, w_tensor], dim=1)))
    #     c = nn.LeakyReLU()(weight_layer(torch.cat([target_embeddings.repeat(1,self.num_neighbor,1), neighbor_embeddings], dim=1) * p))
    #     a = nn.Softmax()(torch.reshape(c, (-1, self.num_neighbor)))
    #     a_reshape = torch.reshape(a, (-1, self.num_neighbor, 1))
    #     out = torch.reshape(torch.sum(hidden_neighbor * a_reshape, dim=1), (-1, self.embedding_dim, 1)) # bacth_size, dim, 1
    #     return out

    # def multi_head_attention(self, neighbor_embeddings, target_embeddings, t_tensor, w_tensor, hidden_layer, weight_layer):
    #     neighbor_embeddings = neighbor_embeddings.cpu()
    #     t_tensor = t_tensor.cpu()
    #     w_tensor = w_tensor.cpu()
        
    #     hidden_neighbor = hidden_layer(neighbor_embeddings)
    #     hidden_target = hidden_layer(target_embeddings)
    #     # tw_cat = torch.cat([t_tensor, w_tensor], dim=1)
    #     tw_cat = torch.cat([t_tensor, w_tensor], dim=2)
    #     print('tshape: {}, wshape: {}, tw_cat shape: {}'.format(t_tensor.shape, w_tensor.shape, tw_cat.shape))
    #     p = nn.Sigmoid()(self.attention_p(tw_cat))
    #     # his_neighbor_head_1 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_1, self.attention_weight_his_1)
    #     # 2*256 1
    #     tt_reshape = target_embeddings.repeat(1, self.num_neighbor, 1)
    #     print('t shape: {}'.format(target_embeddings.shape))
    #     print('0: {}, 2: {}'.format(target_embeddings.repeat(self.num_neighbor, 1, 1).shape, target_embeddings.repeat(1, 1, self.num_neighbor).shape))
    #     print('tt_reshape shape: {}, neighbor_emb shape: {}'.format(tt_reshape.shape, neighbor_embeddings.shape))
    #     # tn_cat = torch.cat([tt_reshape, neighbor_embeddings], dim=1)
    #     tn_cat = torch.cat([tt_reshape, neighbor_embeddings], dim=2)
    #     tn_p = tn_cat * p
    #     print('tn_cat shape: {}, multiply shape: {}'.format(tn_cat.shape, tn_p.shape))
    #     w = weight_layer(tn_p)
    #     print('wshape: {}'.format(w.shape))
    #     # c = nn.LeakyReLU()(weight_layer(torch.cat([target_embeddings.repeat(1,num_neighbors,1), neighbor_embeddings], dim=1) * p))
    #     c = nn.LeakyReLU()(w)
    #     a = nn.Softmax()(torch.reshape(c, (-1, self.num_neighbor)))
    #     a_reshape = torch.reshape(a, (-1, self.num_neighbor, 1))
    #     out = torch.reshape(torch.sum(hidden_neighbor * a_reshape, dim=1), (-1, self.embedding_dim, 1)) # bacth_size, dim, 1
    #     return out

    def his_neighbor_attention(self, neighbor_embeddings, target_embeddings, t_tensor, w_tensor):
        his_neighbor_head_1 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_1, self.attention_weight_his_1)
        his_neighbor_head_2 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_2, self.attention_weight_his_2)
        his_neighbor_head_3 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_3, self.attention_weight_his_3)
        his_neighbor_embedding = torch.mean(torch.cat([his_neighbor_head_1, his_neighbor_head_2, his_neighbor_head_3], dim=2), dim=2) # batch_size, dim
        return his_neighbor_embedding

    def com_neighbor_attention(self, neighbor_embeddings, target_embeddings, t_tensor, w_tensor):
        com_neighbor_head_1 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_1, self.attention_weight_com_1)
        com_neighbor_head_2 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_2, self.attention_weight_com_2)
        com_neighbor_head_3 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_3, self.attention_weight_com_3)
        # com_neighbor_embedding = torch.mean(torch.cat([his_neighbor_head_1, his_neighbor_head_2, his_neighbor_head_3], dim=2), dim=2) # batch_size, dim
        com_neighbor_embedding = torch.mean(torch.cat([com_neighbor_head_1, com_neighbor_head_2, com_neighbor_head_3], dim=2), dim=2) # batch_size, dim
        return com_neighbor_embedding

    def seq_neighbor_attention(self, neighbor_embeddings, target_embeddings, t_tensor, w_tensor):
        seq_neighbor_head_1 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_1, self.attention_weight_seq_1)
        seq_neighbor_head_2 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_2, self.attention_weight_seq_2)
        seq_neighbor_head_3 = self.multi_head_attention(neighbor_embeddings, target_embeddings, t_tensor, w_tensor, self.attention_hidden_3, self.attention_weight_seq_3)
        seq_neighbor_embedding = torch.mean(torch.cat([seq_neighbor_head_1, seq_neighbor_head_2, seq_neighbor_head_3], dim=2), dim=2) # batch_size, dim
        return seq_neighbor_embedding

    def self_attention(self, hidden_his_neighbor, hidden_com_neighbor, hidden_seq_neighbor):
        h = torch.cat([torch.reshape(hidden_his_neighbor, (-1, 1, self.embedding_dim)), torch.reshape(hidden_com_neighbor, (-1, 1, self.embedding_dim)), torch.reshape(hidden_seq_neighbor, (-1, 1, self.embedding_dim))], dim = 1)
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        # attention_scores = attention_scores/torch.sqrt(self.num_self_attention_head)
        attention_scores = attention_scores/np.sqrt(self.num_self_attention_head)
        attention_prob = nn.Softmax()(attention_scores)
        context_layer = torch.reshape(torch.matmul(attention_prob, v), [-1, 3*self.num_self_attention_head])
        out = nn.Softmax()(self.fc(context_layer)) # batch_size, dim
        return out


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

def initialize_neighbors_dicts():
    global his_user2item, his_item2user, com_user2user, com_item2item
    # global last_his_user2item, last_his_item2user
    his_user2item = defaultdict(list)
    his_item2user = defaultdict(list)
    com_user2user = defaultdict(list)
    com_item2item = defaultdict(list)
#     seq_user2user = defaultdict(list)
#     seq_item2item = defaultdict(list)

def get_max_neighbor_tbatchid(nodeid, neighbor_dict, batchid_dict):
    if nodeid not in neighbor_dict:
        return -1
    else:
        return max([batchid_dict[r[0]] for r in neighbor_dict[nodeid]])

# LOCAL RELATION GRAPH CONSTRUCTION
def get_historical_neighbor(sorc_id, dest_id, t, his_dict, args):
    if sorc_id not in his_dict:
        his_dict[sorc_id] = [(dest_id, t, 1)]
    else:
        neighbor_nodes = [relation[0] for relation in his_dict[sorc_id]]
        if dest_id in neighbor_nodes:
            idx = neighbor_nodes.index(dest_id)
            his_dict[sorc_id][idx] = (dest_id, t, his_dict[sorc_id][idx][2]+1)
        else:
            if (len(neighbor_nodes)) >= args.num_neighbor:
                neighbor_nodes_sort = sorted(his_dict[sorc_id], key=lambda x: x[1], reverse=True)
                pop_id = neighbor_nodes_sort.pop()
                his_dict[sorc_id] = neighbor_nodes_sort
            his_dict[sorc_id].append((dest_id, t, 1))
    return his_dict

def get_common_neighbor(sorc_id, dest_id, current_timestamp, delta_T, com_dict, T_sequence, args):
    for i in range(len(T_sequence)-1, -1, -1):
        each_t = T_sequence[i][1]
        if current_timestamp-each_t > 0 and current_timestamp-each_t < delta_T:
            dest_node = T_sequence[i][0]
            com_dict = get_common_neighbor_each(sorc_id, dest_id, current_timestamp, com_dict, args)
            com_dict = get_common_neighbor_each(dest_id, sorc_id, current_timestamp, com_dict, args)
        elif current_timestamp-each_t >= delta_T:
            break
    return com_dict

# def get_common_neighbor_each(sorc_id, dest_id, current_timestamp, com_dict):
def get_common_neighbor_each(sorc_id, dest_node, current_timestamp, com_dict, args):
    if sorc_id not in com_dict:
        com_dict[sorc_id] = [(dest_node, current_timestamp, 1)]
    else:
        neighbor_nodes = [r[0] for r in com_dict[sorc_id]]
        if dest_node in neighbor_nodes:
            idx = neighbor_nodes.index(dest_node)
            com_dict[sorc_id][idx] = (dest_node, current_timestamp, com_dict[sorc_id][idx][2]+1)
        else:
            if (len(neighbor_nodes)) >= args.num_neighbor:
                neighbor_nodes_sort = sorted(com_dict[sorc_id], key=lambda x: x[1], reverse=True)
                pop_id = neighbor_nodes_sort.pop()
                com_dict[sorc_id] = neighbor_nodes_sort
            com_dict[sorc_id].append((dest_node, current_timestamp, 1))
    return com_dict

def get_relation_neighbor_embeddings(target_nodes, neighbor_dict, node_embeddings, zero_neighbor_embedding, zero_weight_embedding, args):
    for i in range(len(target_nodes)):
        each_node = target_nodes[i]
        if each_node not in neighbor_dict:
            each_neighbor_emb = zero_neighbor_embedding
            each_t_emb = zero_weight_embedding
            each_w_emb = zero_weight_embedding
        else:
            # node_list = torch.LongTensor([r[0] for r in neighbor_dict[each_node]]).cpu()
            # t_list = torch.LongTensor([r[1] for r in neighbor_dict[each_node]]).cpu()
            # w_list = torch.LongTensor([r[2] for r in neighbor_dict[each_node]]).cpu()
            node_list = torch.FloatTensor([r[0] for r in neighbor_dict[each_node]])
            t_list = torch.FloatTensor([r[1] for r in neighbor_dict[each_node]])
            w_list = torch.FloatTensor([r[2] for r in neighbor_dict[each_node]])
            long_node_list = torch.LongTensor(node_list.cpu().numpy())
            # long_t_list = torch.LongTensor(t_list.cpu().numpy()).cpu()
            # long_w_list = torch.LongTensor(w_list.cpu().numpy()).cpu()
            each_neighbor_emb_cat = torch.cat([torch.reshape(node_embeddings.cpu()[long_node_list,:], (1, -1)), torch.reshape(zero_neighbor_embedding, (1,-1))], dim = 1)
            each_t_emb_cat = torch.cat([torch.reshape(t_list, (1, -1)), torch.reshape(zero_weight_embedding, (1,-1))], dim = 1)
            each_w_emb_cat = torch.cat([torch.reshape(w_list, (1, -1)), torch.reshape(zero_weight_embedding, (1,-1))], dim = 1)
            each_neighbor_emb = torch.reshape(each_neighbor_emb_cat[:,:args.num_neighbor*args.embedding_dim], (1, args.num_neighbor, args.embedding_dim))
            each_t_emb = torch.reshape(each_t_emb_cat[:,:args.num_neighbor], (1, args.num_neighbor, 1))
            each_w_emb = torch.reshape(each_w_emb_cat[:,:args.num_neighbor], (1, args.num_neighbor, 1))
        if i == 0:
          relation_neighbor_emb = each_neighbor_emb
          relation_t_emb = each_t_emb
          relation_w_emb = each_w_emb
        else:
          relation_neighbor_emb = torch.cat([relation_neighbor_emb, each_neighbor_emb], dim = 0)
          relation_t_emb = torch.cat([relation_t_emb, each_t_emb], dim = 0)
          relation_w_emb = torch.cat([relation_w_emb, each_w_emb], dim = 0)
    return relation_neighbor_emb, relation_t_emb, relation_w_emb

def get_seq_neighbor_embeddings(target_nodes, his_neighbor_emb_input, node_embeddings_input, zero_neighbor_embedding, zero_weight_embedding, args):
    node_seq_embedding = torch.mean(his_neighbor_emb_input, dim=1) # batch_size, dim
    zero = torch.zeros([list(node_seq_embedding.size())[0], 1], dtype=torch.float)
    for i in range(len(target_nodes)):
        each_node_embedding = node_seq_embedding[i]
        each_node_embedding = each_node_embedding.expand(len(target_nodes), args.embedding_dim)
        # cos = F.CosineSimilarity(each_node_embedding, node_seq_embedding, dim=1)
        cos = F.cosine_similarity(each_node_embedding, node_seq_embedding, dim=1)
        cos_reshape = torch.reshape(cos, (-1,1))
        mask = torch.where(cos_reshape > 0.5, cos_reshape, zero)
        mask_index = torch.reshape(torch.nonzero(mask), (1,-1))
        each_node_neighbor_embedding = node_embeddings_input[mask_index, :] #n,dim
        each_w = cos_reshape[mask_index, :] #n,1

        each_neighbor_emb_cat = torch.cat([torch.reshape(each_node_neighbor_embedding.cpu(), (1, -1)), torch.reshape(zero_neighbor_embedding, (1,-1))], dim = 1)
        each_t_emb_cat = torch.cat([torch.reshape(each_w, (1, -1)), torch.reshape(zero_weight_embedding, (1,-1))], dim = 1)
        each_w_emb_cat = torch.cat([torch.reshape(each_w, (1, -1)), torch.reshape(zero_weight_embedding, (1,-1))], dim = 1)
        each_neighbor_emb = torch.reshape(each_neighbor_emb_cat[:,:args.num_neighbor*args.embedding_dim], (1, args.num_neighbor, args.embedding_dim))
        each_t_emb = torch.reshape(each_t_emb_cat[:,:args.num_neighbor], (1, args.num_neighbor, 1))
        each_w_emb = torch.reshape(each_w_emb_cat[:,:args.num_neighbor], (1, args.num_neighbor, 1))
        if i == 0:
          relation_neighbor_emb = each_neighbor_emb
          relation_t_emb = each_t_emb
          relation_w_emb = each_w_emb
        else:
          relation_neighbor_emb = torch.cat([relation_neighbor_emb, each_neighbor_emb], dim = 0)
          relation_t_emb = torch.cat([relation_t_emb, each_t_emb], dim = 0)
          relation_w_emb = torch.cat([relation_w_emb, each_w_emb], dim = 0)
    return relation_neighbor_emb, relation_t_emb, relation_w_emb

# CALCULATE LOSS FOR THE PREDICTED USER STATE
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    # y = Variable(torch.LongTensor(y_true).cpu()[tbatch_interactionids])
    y = Variable(torch.FloatTensor(y_true).cpu()[tbatch_interactionids])

    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series=None, item_embeddings_time_series=None, his_user2item=defaultdict(list), his_item2user=defaultdict(list), com_user2user=defaultdict(list), com_item2item=defaultdict(list), path=PATH):
    print ("*** Saving embeddings and model ***")
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

    # add neighbor dict
    state['his_user2item'] = his_user2item.data.cpu().numpy()
    state['his_item2user'] = his_item2user.data.cpu().numpy()
    state['com_user2user'] = com_user2user.data.cpu().numpy()
    state['com_item2item'] = com_item2item.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s' % args.network)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.model, epoch, args.train_proportion))
    torch.save(state, filename)
    print ("*** Saved embeddings and model to file: %s ***\n\n" % filename)


# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):
    modelname = args.model
    filename = PATH + "saved_models/%s/checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.network, modelname, epoch, args.train_proportion)
    checkpoint = torch.load(filename)
    print ("Loading saved embeddings and model: %s" % filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cpu())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cpu())
    try:
        train_end_idx = checkpoint['train_end_idx']
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cpu())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cpu())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    global his_user2item, his_item2user, com_user2user, com_item2item
    try:
        his_user2item = torch.from_numpy(checkpoint['his_user2item']).cpu()
        his_item2user = torch.from_numpy(checkpoint['his_item2user']).cpu()
        com_user2user = torch.from_numpy(checkpoint['com_user2user']).cpu()
        com_item2item = torch.from_numpy(checkpoint['com_item2item']).cpu()
    except:
        his_user2item = defaultdict(list)
        his_item2user = defaultdict(list)
        com_user2user = defaultdict(list)
        com_item2item = defaultdict(list)

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

