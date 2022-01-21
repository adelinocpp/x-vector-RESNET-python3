"""
Created on Fri Sep  3 14:36:59 2021

@author: adelino
"""
import configure as c
from DB_wav_reader import find_xvector
# import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
# from utils.SpheringSVD import SpheringSVD as Sphering
# from utils.PLDA import plda as PLDA
import plda
import pickle
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde

def trap_int(x,y):
    npts = len(y)
    z = np.zeros((npts,))
    z[1:-1] = z[1:-1] + y[0:-2]
    z[1:-1] = z[1:-1] + y[1:-1]
    z = 0.5*(x[1] - x[0])*z
    return np.cumsum(z)


# use_cuda = False
embedding_size = 512
show_result = True
# n_classes = 104
file_list = find_xvector(c.TEST_VECTOR_DIR)
file_list.sort()
embeddings = {}
numEnroll = len(file_list)
X = np.zeros((numEnroll,embedding_size))
y = np.zeros((numEnroll,))
for i, file in enumerate(file_list):
    spk = int(file.split('/')[-1].split('_')[1]) # filename: DIR_OF_FILE/L_####_U_####_.pth
    y[i] = spk;
    # print("i,spk ({:d},{:d})".format(i,spk))
    with open(file, "rb") as fp: 
        embeddings[spk] = pickle.load(fp)
        fp.close()
    # embeddings[spk] = torch.load(file)
    X[i,:] = np.array(embeddings[spk])
print('Carregados {:03d} x-vectors'.format(len(embeddings)))
 
Xsph = X
# --- processo LDA -------------------------------------------------------
LDAFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.LDA_FILE)
if os.path.exists(LDAFileName):
    with open(LDAFileName,'rb') as fp:
        LDAModel = pickle.load(fp)
        fp.close()
        
Xlda = LDAModel.transform(Xsph)
# --- processo PLDA -------------------------------------------------------
PLDAFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.PLDA_FILE)
if os.path.exists(PLDAFileName):
    with open(PLDAFileName,'rb') as fp:
        PLDAModel = pickle.load(fp)
        fp.close()
        
# --- Carrega os dados de treinamento -------------------------------------
TRAIN_THR_FileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.CALIBRATE_THR_FILE)
with open(TRAIN_THR_FileName,'rb') as fp:
    eer_threshold = pickle.load(fp)
    center_threshold = pickle.load(fp)
    pond_threshold = pickle.load(fp)
    fp.close()

# --- Matriz de resultados ------------------------------------------------
MTXFileName = os.path.join(c.TEST_RESULTS_DIR, c.TEST_CONF_MTX)
U_model = PLDAModel.model.transform(Xlda, from_space='D', to_space='U_model')
hEnroll = math.ceil(0.5*numEnroll)
mtxScore = np.zeros((hEnroll,hEnroll))
y_deff = np.zeros((hEnroll*hEnroll))
y_pred = np.zeros((hEnroll*hEnroll))
iK = 0;
k = 0;
for i in range(0,numEnroll,2):
    U_datum_0 = U_model[i][None,]
    jK = 0;
    for j in range(1,numEnroll,2):
        U_datum_1 = U_model[j][None,]
        log_ratio_0_1 = PLDAModel.model.calc_same_diff_log_likelihood_ratio(U_datum_0, U_datum_1)
        mtxScore[iK,jK] = log_ratio_0_1
        print("Scorring ({:03d},{:03d}) ({:06d})".format(i,j,i*numEnroll + j))
        y_deff[k] = int(y[i] == y[j])
        y_pred[k] = mtxScore[iK,jK] 
        jK += 1
        k += 1
    iK += 1

idxSS = np.nonzero(y_deff == 1)
idxDS = np.nonzero(y_deff == 0)
SSscore = y_pred[idxSS]
DSscore = y_pred[idxDS]
        
with open(MTXFileName,'wb') as fp:
    pickle.dump(mtxScore, fp, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_deff, fp, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_pred, fp, protocol=pickle.HIGHEST_PROTOCOL)
    fp.close()

if (show_result):           	
    densSS = gaussian_kde(SSscore)
    densDS = gaussian_kde(DSscore)
    xss_vals = np.linspace(np.min(y_pred),np.max(y_pred) + np.std(y_pred),10000) 
    xds_vals = np.linspace(np.min(y_pred),np.max(y_pred) + np.std(y_pred),10000) 
    pdfSS = densSS.pdf(xss_vals)
    pdfDS = densDS.pdf(xds_vals)

    cdfSS = trap_int(xss_vals,pdfSS)
    cdfDS = trap_int(xds_vals,pdfDS)

    fig = plt.figure(figsize=(10,8))
    plt.plot(xss_vals,cdfSS)
    plt.plot(xds_vals,1-cdfDS)
    plt.xlabel('score')
    plt.ylabel('acum prob.')
#    plt.ylim(0, max(xss_vals)) # consistent scale
    plt.xlim((np.mean(DSscore) - 1.96*np.std(DSscore)), (np.mean(SSscore) + 1.96*np.std(SSscore))) # consistent scale
    plt.grid(True)
    # plt.show()
    fig.savefig('TEST_plot_00.png', bbox_inches='tight')

    fig = plt.figure(figsize=(10,8))
    plt.plot(xss_vals,pdfSS)
    plt.plot(xds_vals,pdfDS)
    plt.xlabel('score')
    plt.ylabel('prob.')
#    plt.ylim(0, max(SSscore)) # consistent scale
    plt.xlim((np.mean(DSscore) - 1.96*np.std(DSscore)), (np.mean(SSscore) + 1.96*np.std(SSscore))) # consistent scale
    plt.grid(True)
    # plt.show()
    fig.savefig('TEST_plot_01.png', bbox_inches='tight')    


fpr, tpr, threshold = roc_curve(y_deff, y_pred, pos_label=1)
fnr = 1 - tpr
idx = np.nanargmin(np.absolute((fnr - fpr)))
eer_threshold = threshold[idx]
print("fpr: {:5.3f}, tpr: {:5.3f}, fnr: {:5.3f}, thr: {:5.3f}".format(fpr[idx],tpr[idx],fnr[idx],eer_threshold ))
center_threshold = 0.5*(np.max(DSscore) + np.min(SSscore))
pond_threshold = (len(idxSS[0])*np.max(DSscore) + len(idxDS[0])*np.min(SSscore))/len(y_deff)

# print("Scores - SS min: {:5.3f} DS max: {:5.3f}".format(np.min(SSscore),np.max(DSscore)))
# print("Threshoold - eer: {:5.3f} center: {:5.3f} ponder: {:5.3f}".format(eer_threshold,center_threshold,pond_threshold ))
# TRAIN_THR_FileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.CALIBRATE_THR_FILE)
# if not os.path.exists(TRAIN_THR_FileName):
#     with open(TRAIN_THR_FileName,'wb') as fp:
#         pickle.dump(eer_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
#         pickle.dump(center_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
#         pickle.dump(pond_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
#         fp.close()

