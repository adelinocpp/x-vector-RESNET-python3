"""
Created on Wed Aug 18 09:38:54 2021

@author: adelino
"""
import os
import configure as c
import librosa
import numpy as np
from python_speech_features import mfcc, delta, logfbank
from DB_wav_reader import find_wavs, find_feats
import pickle # For python3 
from utils import vad, welford
import math

# ----------------------------------------------------------------------------
def normalize_frames(m,ubm_mean,ubm_std):
    return (m - ubm_mean) / (ubm_std + 2e-12)
# ----------------------------------------------------------------------------
def compute_features(filename,sample_rate,use_Scale, win_length=0.025, \
                     hop_length=0.01, n_mels=13, computeubmstats=True):

    n_win_length = math.ceil(sample_rate*win_length)
    # n_hop_length = math.ceil(sample_rate*hop_length)
    n_FFT = 2 ** math.ceil(math.log2(n_win_length))
    
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    
    vad_sohn = vad.VAD(audio, sr, nFFT=n_FFT, win_length=win_length, \
                       hop_length=hop_length, theshold=0.7)
    mfcc_feat = mfcc(audio, samplerate=sr, nfft=n_FFT, numcep=13, nfilt=26,\
                    winstep=hop_length, winlen=win_length, winfunc=np.hamming)
    mfcc_delta = delta(mfcc_feat,2)    
    mfcc_delta2 = delta(mfcc_delta,2)
    
    fbank_feat = logfbank(audio, samplerate=sr, nfft=n_FFT, nfilt=26,\
                    winstep=hop_length, winlen=win_length)
    
    fbank_delta = delta(fbank_feat,2)    
    fbank_delta2 = delta(fbank_delta,2)
    mfcc_feat = np.concatenate((mfcc_feat, mfcc_delta, mfcc_delta2,fbank_feat,fbank_delta,fbank_delta2),axis=1)
    
    if not (mfcc_feat.shape[0] == vad_sohn.shape[0]):
        if (mfcc_feat.shape[0] > vad_sohn.shape[0]):
            mfcc_feat = mfcc_feat[0:vad_sohn.shape[0],:]
        else:
            vad_sohn = vad_sohn[0:mfcc_feat.shape[0],:]
    
    vadIDX = (vad_sohn == 1).nonzero()[0]
    mfcc_feat = mfcc_feat[vadIDX,:] 
    
    # filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, \
    #                                winlen=0.025, winfunc=np.hamming)    
    # filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
    # feature = normalize_frames(filter_banks, Scale=use_Scale)
    
    return mfcc_feat
# ----------------------------------------------------------------------------

# use_cuda = False # use gpu or cpu
Compute_Train_UBM = True
Normalise_UBM = True
Compute_Norm_Test = True

if (Compute_Train_UBM):
    w_ubm = welford.Welford()
    file_list = find_wavs(c.TRAIN_WAV_DIR)
    splitDir = c.TRAIN_FEAT_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
    print('Inicio do calculo de características:')
    print('Arquivos de treinamento: {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)));
        feat_and_label = {}
        features = compute_features(filename,c.SAMPLE_RATE,c.USE_SCALE)
        if (features.shape[0] < c.NUM_WIN_SIZE):
            print('Finalizado arquivo {:4} de {:4} - Tamanho inadequado'.format(idx, len(file_list)-1));
            continue
        
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.TRAIN_FEAT_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        w_ubm.update(features)
        feat_and_label['feat'] = features
        feat_and_label['label'] = filenameFolder
        if (not os.path.exists(c.TRAIN_FEAT_DIR + '/' + filenameFolder)):
            os.mkdir(c.TRAIN_FEAT_DIR + '/' + filenameFolder)
        with open(filenameSave, 'wb') as f:
            pickle.dump(feat_and_label,f)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        
    ubm_data = {}
    ubm_data["mean"]    = w_ubm.mean
    ubm_data["std"]     = w_ubm.std

    with open(c.UBM_FILE_NAME, 'wb') as f:
        pickle.dump(ubm_data,f)
else:
    with open(c.UBM_FILE_NAME, 'rb') as f:
        ubm_data = pickle.load(f)

if (Normalise_UBM):
    print('Normalização das características de treinamento...')
    file_list = find_feats(c.TRAIN_FEAT_DIR)
    print('Arquivos para normalizar {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)-1));
        with open(filename, 'rb') as f:
            feat_and_label = pickle.load(f)
        feat_and_label['feat'] = normalize_frames(feat_and_label['feat'],\
                                        ubm_data["mean"],ubm_data["std"])
        with open(filename, 'wb') as f:
            pickle.dump(feat_and_label,f)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));

if (Compute_Norm_Test):
    splitDir = c.TEST_FEAT_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
    print('Inicio do calculo de características:')
    file_list = find_wavs(c.TEST_WAV_DIR)
    print('Arquivos de teste {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)));
        feat_and_label = {}
        features = compute_features(filename,c.SAMPLE_RATE,c.USE_SCALE) 
        if (features.shape[0] < c.NUM_WIN_SIZE):
            print('Finalizado arquivo {:4} de {:4} - Tamanho inadequado'.format(idx, len(file_list)-1));
            continue
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.TEST_FEAT_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        feat_and_label['feat'] = normalize_frames(features,\
                                        ubm_data["mean"],ubm_data["std"])
        feat_and_label['label'] = filenameFolder
        if (not os.path.exists(c.TEST_FEAT_DIR + '/' + filenameFolder)):
            os.mkdir(c.TEST_FEAT_DIR + '/' + filenameFolder)
        with open(filenameSave, 'wb') as f:
            pickle.dump(feat_and_label,f)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
# ----------------------------------------------------------------------------            
