'''
This code trains the JODIE model for the given dataset. 
The task is: interaction prediction.

How to run: 
$ python jodie.py --network reddit --model jodie --epochs 50

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

import time

from library_data import *
import library_models as lib
from library_models import *

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
args = parser.parse_args()

args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
if args.gpu == -1:
    args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# MULTI-HEAD ATTENTION
num_heads = 3
dropout = 0.3

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error. 
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500

# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users, num_items, num_heads, dropout).cuda()
weight = torch.Tensor([1,true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # 均匀分布 the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings
user_neighbor_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
item_neighbor_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)

# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_item = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_item_timediffs = {}
cached_tbatches_previous_item = {}

with trange(args.epochs) as progress_bar1:
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

        epoch_start_time = time.time()
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        optimizer.zero_grad() # 归零梯度
        reinitialize_tbatches() # batch 重新划分
        total_loss, loss, total_interaction_count = 0, 0, 0  # 归零 loss

        initialize_neighbors() # 邻居节点dict重新计算
        T = 500

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        with trange(train_end_idx) as progress_bar2:
            for j in progress_bar2:
                progress_bar2.set_description('Processed %dth interactions' % j)

                # READ INTERACTION J
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                feature = feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]
                t = timestamp_sequence[j]
                
                # neighbor nodes
                # historical interaction relations
                his_user2item = get_historical_neighbor(userid, itemid, t, his_user2item) # todo 四个dict是全局变量
                his_item2user = get_historical_neighbor(itemid, userid, t, his_item2user)
                # common interaction relations
                # get sequence
                T_user_sequence = sorted(his_item2user[itemid], key=lambda x: x[1]) # 格式为 [(id,t,w)]
                com_user2user = get_common_neighbor(userid, t, T, com_user2user, T_user_sequence)
                T_item_sequence = sorted(his_user2item[userid], key=lambda x: x[1]) # 格式为 [(id,t,w)]
                com_item2item = get_common_neighbor(itemid, t, T, com_item2item, T_item_sequence)
                batch_idx = t%tbatch_timespan
                if batch_idx > origin_batch_idx:
                    seq_user2user = get_seq_sim_neighbor("user", batch_idx, seq_user2user)
                    seq_item2item = get_seq_sim_neighbor("item", batch_idx, seq_item2item)
                origin_batch_idx = batch_idx
                # interaction sequence similarity relations
                # 每个 batch 离线算好 emb 相似的节点

                # 第一个 batch
                if is_first_epoch:
                    # get neighbor max index
                    max_neighbor_idx = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid])
                    max_neighbor_idx = get_max_neighbor_idx(userid, his_user2item, max_neighbor_idx, tbatch_start_time, lib.tbatchid_item)
                    max_neighbor_idx = get_max_neighbor_idx(itemid, his_item2user, max_neighbor_idx, tbatch_start_time, lib.tbatchid_user)
                    max_neighbor_idx = get_max_neighbor_idx(userid, com_user2user, max_neighbor_idx, tbatch_start_time, lib.tbatchid_user)
                    max_neighbor_idx = get_max_neighbor_idx(itemid, com_item2item, max_neighbor_idx, tbatch_start_time, lib.tbatchid_item)
                    max_neighbor_idx = get_max_neighbor_idx(userid, seq_user2user, max_neighbor_idx, tbatch_start_time, lib.tbatchid_user)
                    max_neighbor_idx = get_max_neighbor_idx(itemid, seq_item2item, max_neighbor_idx, tbatch_start_time, lib.tbatchid_item)
                    
                    
                    # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                    tbatch_to_insert = max_neighbor_idx + 1
                    lib.tbatchid_user[userid] = tbatch_to_insert
                    lib.tbatchid_item[itemid] = tbatch_to_insert

                    lib.current_tbatches_user[tbatch_to_insert].append(userid)
                    lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                    lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                    lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                    lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                    lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                    lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

                timestamp = timestamp_sequence[j]
                if tbatch_start_time is None:
                    tbatch_start_time = timestamp

                # 到第二个 batch 时对第一个 batch 进行处理
                # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                if timestamp - tbatch_start_time > tbatch_timespan: # 每个 batch 有时间段限制
                    tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

                    # ITERATE OVER ALL T-BATCHES
                    if not is_first_epoch:
                        lib.current_tbatches_user = cached_tbatches_user[timestamp]
                        lib.current_tbatches_item = cached_tbatches_item[timestamp]
                        lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                        lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                        lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                        lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                        lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]

                    with trange(len(lib.current_tbatches_user)) as progress_bar3:
                        # 遍历一个 batch 中的所有 tbatch，i 是 tbatch 的 index
                        for i in progress_bar3:
                            progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))
                            
                            total_interaction_count += len(lib.current_tbatches_interactionids[i])

                            # LOAD THE CURRENT TBATCH
                            if is_first_epoch:
                                lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                                lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                                lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                                lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).cuda()

                                lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                                lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()
                                lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()

                            tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                            tbatch_itemids = lib.current_tbatches_item[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                            tbatch_interactionids = lib.current_tbatches_interactionids[i]
                            feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                            user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                            item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                            tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                            item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                            #**** user_neighbor_embeddings
                            for user in tbatch_userids:
                                his_neighbor = his_user2item[user]
                                com_neighbor = com_user2user[user]
                                seq_neighbor = seq_user2user[user]
                                his_neighbor_emebddings = []
                                com_neighbor_embeddings = []
                                seq_neighbor_embeddings = []
                                for each_item in his_neighbor:
                                    new_neighbor_emb = model.trans_p(item_embeddings[each_item[0]], each_item[1], each_item[2])
                                    his_neighbor_emebddings.append(new_neighbor_emb)
                                for each_user in com_neighbor:
                                    new_neighbor_emb = model.trans_p(user_embeddings[each_user[0]], each_user[1], each_user[2])
                                    com_neighbor_embeddings.append(new_neighbor_emb)
                                for each_user in seq_neighbor:
                                    new_neighbor_emb = model.trans_p(user_embeddings[each_user[0]], each_user[1], each_user[2])
                                    seq_neighbor_embeddings.append(new_neighbor_emb)
                                neighbor_emb = model.neighbor_aggregate(user_embeddings[user], his_neighbor_emebddings, com_neighbor_embeddings, seq_neighbor_embeddings)
                                user_neighbor_embeddings[user] = neighbor_emb
                            
                            #**** item_neighbor_embeddings
                            for item in tbatch_itemids:
                                his_neighbor = his_item2user[item]
                                com_neighbor = com_item2item[item]
                                seq_neighbor = seq_item2item[item]
                                his_neighbor_emebddings = []
                                com_neighbor_embeddings = []
                                seq_neighbor_embeddings = []
                                for each_user in his_neighbor:
                                    new_neighbor_emb = model.trans_p(user_embeddings[each_user[0]], each_user[1], each_user[2])
                                    his_neighbor_emebddings.append(new_neighbor_emb)
                                for each_item in com_neighbor:
                                    new_neighbor_emb = model.trans_p(item_embeddings[each_item[0]], each_item[1], each_item[2])
                                    com_neighbor_embeddings.append(new_neighbor_emb)
                                for each_item in seq_neighbor:
                                    new_neighbor_emb = model.trans_p(item_embeddings[each_item[0]], each_item[1], each_item[2])
                                    seq_neighbor_embeddings.append(new_neighbor_emb)
                                neighbor_emb = model.neighbor_aggregate(item_embeddings[item], his_neighbor_emebddings, com_neighbor_embeddings, seq_neighbor_embeddings)
                                item_neighbor_embeddings[item] = neighbor_emb

                            # PROJECT USER EMBEDDING TO CURRENT TIME
                            user_embedding_input = user_embeddings[tbatch_userids,:]
                            user_neighbor_input = user_neighbor_embeddings[tbatch_userids,:]
                            item_neighbor_input = item_neighbor_embeddings[tbatch_itemids,:]
                            # user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                            # user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)
                            user_item_embedding = torch.cat([user_embedding_input, user_neighbor_input], dim=1)

                            # PREDICT NEXT ITEM EMBEDDING                            
                            predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                            # CALCULATE PREDICTION LOSS
                            item_embedding_input = item_embeddings[tbatch_itemids,:]
                            # loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())
                            loss += MSELoss(predicted_item_embedding, item_embedding_input.detach())

                            # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                            # user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                            user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=user_neighbor_input, select='user_update')
                            # item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')
                            item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=item_neighbor_input, select='item_update')

                            item_embeddings[tbatch_itemids,:] = item_embedding_output
                            user_embeddings[tbatch_userids,:] = user_embedding_output  

                            user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                            item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                            # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                            loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                            loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                            # CALCULATE STATE CHANGE LOSS
                            if args.state_change:
                                loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss) 

                    # 每个 batch 计算一次 loss，梯度反向传播
                    # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                    user_embeddings.detach_()
                    item_embeddings_timeseries.detach_() 
                    user_embeddings_timeseries.detach_()
                   
                    # REINITIALIZE
                    if is_first_epoch:
                        cached_tbatches_user[timestamp] = lib.current_tbatches_user
                        cached_tbatches_item[timestamp] = lib.current_tbatches_item
                        cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                        cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                        cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                        cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                        cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item
                        
                        reinitialize_tbatches()
                        tbatch_to_insert = -1

        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # END OF ONE EPOCH 
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        # 每个 epoch 初始化 embedding
        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)
