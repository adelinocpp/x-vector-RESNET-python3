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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde


# use_cuda = False
# embedding_size = 512
embedding_size = 128
show_result = True
# n_classes = 104
file_list = find_xvector(c.TRAIN_VECTOR_DIR)
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
    if ((i == 0) and (not embeddings[spk].shape[0] == embedding_size)):
        embedding_size = embeddings[spk].shape[0]
        X = np.zeros((numEnroll,embedding_size))
        
    X[i,:] = np.array(embeddings[spk])
print('Carregados {:03d} x-vectors'.format(len(embeddings)))

# --- processo de Sphearing ----------------------------------------------
# SpheringFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.SPHERING_FILE)
# if os.path.exists(SpheringFileName):
#     with open(SpheringFileName,'rb') as fp:
#         SpheModel = pickle.load(fp)
#         fp.close()
# else:
#     SpheModel = Sphering.SpheringSVD()
#     SpheModel.fit(X)
#     with open(SpheringFileName,'wb') as fp:
#         pickle.dump(SpheModel, fp, protocol=pickle.HIGHEST_PROTOCOL)
#         fp.close()

# Xsph = SpheModel.transform(X)
Xsph = X
# --- processo LDA -------------------------------------------------------
LDAFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.LDA_FILE)
if os.path.exists(LDAFileName):
    with open(LDAFileName,'rb') as fp:
        LDAModel = pickle.load(fp)
        fp.close()
else:
    LDAModel = LinearDiscriminantAnalysis()
    LDAModel.fit(Xsph, y)
    with open(LDAFileName,'wb+') as fp:
        pickle.dump(LDAModel, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()
        
Xlda = LDAModel.transform(Xsph)
# --- processo PLDA -------------------------------------------------------
PLDAFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.PLDA_FILE)
if os.path.exists(PLDAFileName):
    with open(PLDAFileName,'rb') as fp:
        PLDAModel = pickle.load(fp)
        fp.close()
else:
    PLDAModel = plda.Classifier()
    PLDAModel.fit_model(Xlda, y)
    # PLDAModel = PLDA.plda(niter = 1000)
    # PLDAModel.fit(Xlda.T, y)
    with open(PLDAFileName,'wb+') as fp:
        pickle.dump(PLDAModel, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()
# --- Matriz de calibracao ------------------------------------------------
MTXFileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.CALIBRATE_MTX_FILE)
if os.path.exists(MTXFileName):
    with open(MTXFileName,'rb') as fp:
        mtxScore = pickle.load(fp)
        y_deff = pickle.load(fp)
        y_pred = pickle.load(fp)
        fp.close()
else:
    U_model = PLDAModel.model.transform(Xlda, from_space='D', to_space='U_model')
    mtxScore = np.zeros((numEnroll,numEnroll))
    y_deff = np.zeros((numEnroll*numEnroll,))
    y_pred = np.zeros((numEnroll*numEnroll,))
    # k = 0;
    for i in range(0,numEnroll):
        U_datum_0 = U_model[i][None,]
        for j in range (0,numEnroll):
            U_datum_1 = U_model[j][None,]
            log_ratio_0_1 = PLDAModel.model.calc_same_diff_log_likelihood_ratio(U_datum_0, U_datum_1)
            mtxScore[i,j] = log_ratio_0_1
            # mtxScore[i,j] = PLDAModel.score(Xlda[i,:],Xlda[j,])
            print("Scorring ({:03d},{:03d}) ({:06d})".format(i,j,i*numEnroll + j))
            y_deff[i*numEnroll + j] = int(y[i] == y[j])
            y_pred[i*numEnroll + j] = mtxScore[i,j]
    with open(MTXFileName,'wb+') as fp:
        pickle.dump(mtxScore, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_deff, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()
        
# plt.matshow(mtxScore)
# plt.show()
if (show_result):
    idxSS = np.nonzero(y_deff == 1)
    idxDS = np.nonzero(y_deff == 0)
    SSscore = y_pred[idxSS]
    DSscore = y_pred[idxDS]
    
    densSS = gaussian_kde(SSscore)
    densDS = gaussian_kde(DSscore)
    xss_vals = np.linspace(np.min(SSscore),np.max(SSscore),2000) 
    xds_vals = np.linspace(np.min(DSscore),np.max(DSscore),2000) 
     
    pdfSS = densSS.pdf(xss_vals)
    pdfDS = densDS.pdf(xds_vals)
   
#    densSS._compute_covariance()
#    densDS._compute_covariance()

    fig = plt.figure(figsize=(10,8))
    plt.plot(xss_vals,pdfSS)
    plt.plot(xds_vals,pdfDS)
#    plt.plot(xss_vals,densSS(xss_vals))
#    plt.plot(xds_vals,densDS(xds_vals))
    plt.xlabel('score')
    plt.ylabel('prob.')
#    plt.ylim(0, max(SSscore)) # consistent scale
    plt.xlim((np.mean(DSscore) - 1.96*np.std(DSscore)), (np.mean(SSscore) + 1.96*np.std(SSscore))) # consistent scale
    plt.grid(True)
    # plt.show()
    fig.savefig('TRAIN_plda_plot.png', bbox_inches='tight')    

fpr, tpr, threshold = roc_curve(y_deff, y_pred, pos_label=1)
fnr = 1 - tpr
idx = np.nanargmin(np.absolute((fnr - fpr)))
eer_threshold = threshold[idx]
print("fpr: {:5.3f}, tpr: {:5.3f}, fnr: {:5.3f}, thr: {:5.3f}".format(fpr[idx],tpr[idx],fnr[idx],eer_threshold ))
center_threshold = 0.5*(np.max(DSscore) + np.min(SSscore))
pond_threshold = (len(idxSS[0])*np.max(DSscore) + len(idxDS[0])*np.min(SSscore))/len(y_deff)

print("Scores - SS min: {:5.3f} DS max: {:5.3f}".format(np.min(SSscore),np.max(DSscore)))
print("Threshoold - eer: {:5.3f} center: {:5.3f} ponder: {:5.3f}".format(eer_threshold,center_threshold,pond_threshold ))
TRAIN_THR_FileName = os.path.join(c.LDA_SAVE_MODELS_DIR, c.CALIBRATE_THR_FILE)
if not os.path.exists(TRAIN_THR_FileName):
    with open(TRAIN_THR_FileName,'wb') as fp:
        pickle.dump(eer_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(center_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pond_threshold, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()

