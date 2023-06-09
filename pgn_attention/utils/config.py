# 神经网络配置文件
import torch
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
print(root_path)

# General
hidden_size = 512
dec_hidden_size = 512
embed_size = 512
pointer = True

# Data
max_vocab_size = 20000
model_name = 'pgn_model'
embed_file = root_path + '/data/wv/word2vec_pad.model'
source = 'train'
train_data_path = root_path + '/data/train.txt'
val_data_path = root_path + '/data/dev.txt'
test_data_path = root_path + '/data/test.txt'
stop_word_file = root_path + '/data/stopwords.txt'
losses_path = root_path + '/data/loss.txt'
log_path = root_path + '/data/log_train.txt'
word_vector_model_path = root_path + '/data/wv/word2vec_pad.model'
encoder_save_name = root_path + '/src/saved_model/model_encoder.pt'
decoder_save_name = root_path + '/src/saved_model/model_decoder.pt'
attention_save_name = root_path + '/src/saved_model/model_attention.pt'
reduce_state_save_name = root_path + '/src/saved_model/model_reduce_state.pt'
model_save_path = root_path + '/src/saved_model/pgn_model.pt'
max_enc_len = 300
max_dec_len = 100
truncate_enc = True
truncate_dec = True

min_dec_steps = 30
# 在Greedy Decode的时候设置为50
# max_dec_steps = 50
# 在Beam-search Decode的时候设置为30
max_dec_steps = 30

enc_rnn_dropout = 0.5
enc_attn = True
dec_attn = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0

# Training
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 3
batch_size = 8
# is_cuda = True
is_cuda = False
coverage = True


fine_tune = False
scheduled_sampling = False
weight_tying = False

max_grad_norm = 2.0
DEVICE = "cpu"
LAMBDA = 1

# 设定模型服务的路由地址
model_server_url = "http://0.0.0.0:5000/v1/main_server/"

# Beam search
beam_size = 3
alpha = 0.2
beta = 0.2
gamma = 2000

