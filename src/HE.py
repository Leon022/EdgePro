import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import copy
import time
from gtsrb_net import AlexnetTS_CC
from phe import paillier
from options import args_parser
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

net_d = torch.load('./results/mnist/normal_model/ckpt.pth')
w = net_d['net']
# print(net_d['encryption_time'])
# print(net_d['decryption_time'])
# net = imagenets.resnet101(pretrained=True)
# net.load_state_dict(net_d)
# net.to(args.device)

global_pub_key, global_priv_key = paillier.generate_paillier_keypair()
pub_key = global_pub_key
priv_key = global_priv_key

print('Start Encryption!!!')
enc_start = time.time()
update_w = {}
for k in w.keys():
    update_w[k] = w[k]
    list_w = update_w[k].view(-1).cpu().tolist()
    # encryption
    for i, elem in enumerate(list_w):
        list_w[i] = pub_key.encrypt(elem)
    update_w[k] = list_w

update_w2 = copy.deepcopy(update_w)

enc_end = time.time()
encryption_time = enc_end - enc_start
print('Encryption time:', encryption_time)
#
print('Start Decryption!!!')
dec_start = time.time()
w_new = {}
for k in update_w.keys():
    # decryption
    for i, elem in enumerate(update_w[k]):
        update_w[k][i] = priv_key.decrypt(elem)

    origin_shape = list(w[k].size())
    w_new[k] = torch.FloatTensor(update_w[k]).to(args.device).view(*origin_shape)

dec_end = time.time()
decryption_time = dec_end - dec_start
print('Decryption time:', decryption_time)

state = {
            'net': update_w2,
            'encryption_time': encryption_time,
            'decryption_time': decryption_time,
            }

torch.save(state, './results/mnist/normal_model/ckpt_he.pth')