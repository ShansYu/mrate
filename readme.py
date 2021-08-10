# 参数注释
# 从原数据文件解析出来的各项列表
user_sequence = [] # array
item_sequence = [] # array
label_sequence = []
feature_sequence = []
timestamp_sequence = [] # array
start_timestamp = None
y_true_labels = []

user2id = {}
item2id = {}

item_timedifference_sequence = []  # item 本次交互与上次交互的时间差 # array scale
item_sequence_id = [] # 列表 item_sequence 对应的 id
user_timedifference_sequence = [] # user 本次交互与上次交互的时间差 # array scale
user_sequence_id = [] # 列表 user_sequence 对应的 id

tbatchid_user[userid] = tbatch_to_insert  # the latest tbatch a user is in
tbatchid_item[itemid] = tbatch_to_insert
lib.current_tbatches_user[tbatch_to_insert].append(userid) # dict: tbatch_index -> userid_list
lib.current_tbatches_item[tbatch_to_insert].append(itemid)
lib.current_tbatches_feature[tbatch_to_insert].append(feature)
lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j]) # 用户上一次交互的itemid
