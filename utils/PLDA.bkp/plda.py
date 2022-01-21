# -------------------------------
import numpy as np
import numpy.matlib as mb

# -----------------------------------------------------------------------------
def length_norm(X):
    return X/np.sqrt(sum(np.power(X,2)))
# -----------------------------------------------------------------------------
def calc_white_mat(sigma, eps = 1e-10):
    D,V = np.linalg.svd(sigma)[1:3]
    return np.dot(V, np.diag(1.0/np.sqrt(D + eps))) 
# -----------------------------------------------------------------------------
class plda:
    def __init__(self, niter = 500):
        self.niter = niter
        self.nphi = 0
        self.Phi = 1
        self.Sigma = 1
        self.W1 = 1
        self.M = 0
        
    # ------------------------------------------------------------------------    
    def __expectation_plda(self, data, Phi, Sigma, spk_counts):
        # computes the posterior mean and covariance of the factors
        nphi     = Phi.shape[1]
        nsamples = data.shape[1]
        nspks    = spk_counts.shape[0]

        Ey  = np.zeros([nphi, nsamples])
        Eyy = np.zeros([nphi,nphi])

        # initialize common terms to save computations
        uniqFreqs  	  = np.unique(spk_counts)
        nuniq 		  = uniqFreqs.shape[0]
        invTerms      = [np.zeros([nphi,nphi])]*nuniq
        Sigma_inv = np.linalg.inv(Sigma)
        PhiT_invS_Phi = np.dot(np.dot(Phi.T,Sigma_inv),Phi)
        I = np.eye(nphi);
        for ix in range(0,nuniq):
            nPhiT_invS_Phi = uniqFreqs[ix] * PhiT_invS_Phi
            Cyy =  np.linalg.pinv(I + nPhiT_invS_Phi)
            invTerms[ix] = Cyy

        data2 = np.dot(Sigma_inv,data)
        cnt  = 0;
        for spk in range(0,nspks):
            nsessions = spk_counts[spk]
            # Speaker indices
            idx = np.arange(cnt,cnt + spk_counts[spk])
            cnt  = cnt + spk_counts[spk]
            Data = data2[:, idx];
            PhiT_invS_y = np.sum(np.dot(Phi.T,Data), axis=1)
            PhiT_invS_y.shape = (PhiT_invS_y.shape[0],1)
            indexSel = np.where(uniqFreqs == nsessions)[0][0]
            Cyy = invTerms[indexSel];
            Ey_spk  = np.dot(Cyy ,PhiT_invS_y)
            Eyy_spk = Cyy + np.dot(Ey_spk,Ey_spk.T)
            Eyy     = Eyy + nsessions*Eyy_spk;
            Ey_spk.shape = (Ey_spk.shape[0],1)
            Ey[:, idx] = mb.repmat(Ey_spk, 1, nsessions)
            
        return Ey, Eyy
    # ------------------------------------------------------------------------    
    def __maximization_plda(self, data, Ey, Eyy):
        # ML re-estimation of the Eignevoice subspace and the covariance of the
        # residual noise (full).
        nsamples    = data.shape[1]
        Data_sqr    = np.dot(data,data.T)
        Phi         = np.dot(np.dot(data, Ey.T), np.linalg.inv(Eyy) );
        Sigma       = (1/nsamples)*(Data_sqr - np.dot(np.dot(Phi,Ey), data.T))
        # Sigma       = np.diag(np.diag(Sigma))
        return Phi, Sigma
    # ------------------------------------------------------------------------
    def fit(self,X,y,nphi = 0):
        if (nphi == 0):
            self.nphi = len(np.unique(y)) - 1;
        else:
            self.nphi = max(len(np.unique(y)) - 1,10);        
        if not (X.shape[1] == y.shape[0]):
            print('PLDA error: oh dear! number of data samples should match the number of labels!')
            return        
        ndim = X.shape[0]
        idx = np.argsort(y)
        y = y[idx]
        X = X[:,idx]
        A = np.unique(y)-0.5
        A = np.append(A,max(y) + 0.5)
        spk_counts = np.histogram(y,A)[0]      
        self.M = np.mean(X, axis=1);
        Xc = (X.T - self.M).T
        Xc = length_norm(Xc)
        self.W1 = calc_white_mat(np.cov(Xc, rowvar=True)) 
        Xc = np.dot(self.W1,Xc)      
        # np.random.seed(int(time.time()))
        np.random.seed()
        Sigma    = 100 * np.random.randn(ndim,ndim)  # covariance matrix of the residual term
        # np.random.seed(int(time.time()))
        np.random.seed()
        Phi = np.random.randn(ndim, self.nphi); # factor loading matrix (Eignevoice matrix)
        Phi = (Phi.T - np.mean(Phi, axis=1)).T
        W2   = calc_white_mat(np.dot(Phi.T,Phi));
        Phi = np.dot(Phi,W2);  
        self.Phi   = Phi;
        self.Sigma = Sigma;                 # orthogonalize Eigenvoices (columns)
        # EyAnt = np.zeros((ndim,y.shape[0])) 
        # EyyAnt   = np.zeros((ndim,ndim))
        print('Re-estimating the Eigenvoice subspace with',self.nphi,' factors ...');
        for iter in range(0,self.niter):
            print('EM iter#: ',iter);
            # expectation
            Ey, Eyy = self.__expectation_plda(Xc, Phi, Sigma, spk_counts)
            # diff_1 = np.power(EyAnt - Ey,2).mean()
            # diff_2 = np.power(EyyAnt - Eyy,2).mean()
            # EyAnt = Ey
            # EyyAnt = Eyy
            # print('EM diff expectation:  %4.2e - %4.2e' % (diff_1, diff_2));
            # maximization
            Phi, Sigma = self.__maximization_plda(Xc, Ey, Eyy)
            diff_1 = np.power(self.Phi - Phi,2).mean()
            diff_2 = np.power(self.Sigma - Sigma,2).mean()
            print('EM diff maximization: %4.2e - %4.2e' % (diff_1, diff_2));
            self.Phi   = Phi;
            self.Sigma = Sigma;
    # ------------------------------------------------------------------------
    def score(self, model_iv, test_iv):
        Phi     = self.Phi
        Sigma   = self.Sigma
        W       = self.W1
        M       = self.M
        # --- post-processing the model x-vectors
        model_iv = model_iv - M             # centering the data
        model_iv = length_norm(model_iv)    # normalizing the length
        model_iv = np.dot(W,model_iv)     # whitening data
        model_iv.shape = (model_iv.shape[0],1)
        
        # --- post-processing the test x-vectors 
        test_iv = test_iv - M               # centering the data
        test_iv = length_norm(test_iv)      # normalizing the length
        test_iv  = np.dot(W,test_iv)      # whitening data
        test_iv.shape = (test_iv.shape[0],1)
        
        nphi = Phi.shape[0]

        Sigma_ac  = np.dot(Phi,Phi.T)
        Sigma_tot = Sigma_ac + Sigma

        Sigma_tot_i = np.linalg.pinv(Sigma_tot)
        Sigma_i = np.linalg.pinv(Sigma_tot - np.dot(np.dot(Sigma_ac, Sigma_tot_i),Sigma_ac))
        Q = Sigma_tot_i - Sigma_i
        P = np.dot(np.dot(Sigma_tot_i,Sigma_ac),Sigma_i)
        U, S = np.linalg.svd(P)[0:2]
        
        Lambda = np.diag(S[0:nphi])
        Uk     = U[:,0:nphi]
        Q_hat  = np.dot(Uk.T ,Q ,Uk)
        model_iv = np.dot(Uk.T, model_iv)
        test_iv  = np.dot(Uk.T, test_iv)

        score_h1 = np.dot(np.dot(model_iv.T, Q_hat), model_iv)
        score_h2 = np.dot(np.dot(test_iv.T , Q_hat), test_iv)
        score_h1h2 = 2 * np.dot(np.dot(model_iv.T,Lambda),test_iv)
        scores = score_h1h2 + score_h1
        scores = scores + score_h2.T
        return scores
