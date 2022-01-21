"""
Created on Sat Aug 28 10:20:11 2021

@author: adelino
"""

import torch
# import torch.nn.functional as F
# from torch.autograd import Variable

import pandas as pd
import math
import os
import configure as c

from DB_wav_reader import read_feats_structure, find_feats, get_num_class
from SR_Dataset import read_MFB, ToTensorTestInput, load_model, get_min_loss_model
# from model.model import background_resnet
# from P01_Train_DNN_v0 import get_min_loss_model
import numpy as np
import pickle

def split_enroll_and_test(dataroot_dir):
    DB_all = read_feats_structure(dataroot_dir)
    enroll_DB = pd.DataFrame()
    test_DB = pd.DataFrame()
    
    enroll_DB = DB_all[DB_all['filename'].str.contains('enroll.p')]
    test_DB = DB_all[DB_all['filename'].str.contains('test.p')]
    
    # Reset the index
    enroll_DB = enroll_DB.reset_index(drop=True)
    test_DB = test_DB.reset_index(drop=True)
    return enroll_DB, test_DB

def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename) # input size:(n_frames, n_dims)
    
    tot_segments = math.ceil(len(input)/test_frames) # total number of segments with 'test_frames' 
    activation = 0
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i*test_frames:i*test_frames+test_frames]
            
            TT = ToTensorTestInput()
            temp_input = TT(temp_input) # size:(1, 1, n_dims, n_frames)
    
            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation,_ = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True)
    activation = l2_norm(activation, c.ALPHA_LNORM)
    return activation

def l2_norm(input, alpha):
    input_size = input.size()  # size:(n_frames, dim)
    buffer = torch.pow(input, 2)  # 2 denotes a squared operation. size:(n_frames, dim)
    normp = torch.sum(buffer, 1).add_(1e-10)  # size:(n_frames)
    norm = torch.sqrt(normp)  # size:(n_frames)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output

def enroll_test_spk(use_cuda, test_frames, model, file_list, embedding_dir):
    """
    Output the averaged d-vector for each speaker (enrollment)
    Return the dictionary (length of n_spk)
    """
    n_files = len(file_list) # 10
    # enroll_speaker_list = sorted(set(DB['speaker_id']))
    file_list.sort()
    numEmbeddings = {}
    
    # Aggregates all the activations
    print("Start to aggregate all the d-vectors per enroll speaker")
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
        
    for i in range(n_files):
        filename = file_list[i]
        file_data = filename.split('/')
        # filename = DB['filename'][i]
        spk = int(file_data[-2]) + 1000      
        activation = get_embeddings(use_cuda, filename, model, test_frames)
        # if spk in embeddings:
        #     numEmbeddings[spk] += 1
        #     embeddings[spk] += embeddings[spk] + \
        #         (np.squeeze(activation.numpy()) - embeddings[spk])/numEmbeddings[spk]
        # else:
        #     numEmbeddings[spk] = 1
        #     embeddings[spk] = np.squeeze(activation.numpy())
        
        activation = np.squeeze(activation.numpy())
        if spk in numEmbeddings:
            numEmbeddings[spk] += 1            
        else:
            numEmbeddings[spk] = 1
        embedding_path = os.path.join(embedding_dir, \
                    'S_{:04d}_U_{:04d}_'.format(spk,numEmbeddings[spk])+'.pth')    
        # print("Aggregates the activation (spk : %04d,utterance %04d)" % (spk,numEmbeddings[spk]))
        # torch.save(embeddings[spk], embedding_path)
        # torch.save(activation, embedding_path)
        with open(embedding_path, "wb") as fp: 
            pickle.dump(activation, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
        print("Save the embeddings for (index, speaker, utterance) %04d %03d %02d"\
              % (i, spk,numEmbeddings[spk]))
            
    return 0
    

use_cuda = False
log_dir = c.SAVE_MODELS_DIR
embedding_size = 512
cp_num,__ = get_min_loss_model(log_dir) # Which checkpoint to use?    
training_classes = get_num_class(c.TRAIN_WAV_DIR)
test_frames = 200

model = load_model(use_cuda, log_dir, cp_num, embedding_size, training_classes, c.BACKBONETYPE)
file_list = find_feats(c.TEST_FEAT_DIR)
   
# Perform the enrollment and save the results
enroll_test_spk(use_cuda, test_frames, model, file_list, c.TEST_VECTOR_DIR)
print('Calculados...')
   
