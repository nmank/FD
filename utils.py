import numpy as np


def chordal_distance(X, Y, Bs_x, Bs_y, manifold = 'Flag'):

    k = len(Bs_x)

    dist = 0

    for i in range(k):
        id_x = Bs_x[i]
        id_y = Bs_y[i]
        Xi = X[:,id_x]
        Yi = Y[:,id_y]
        mm = np.min([len(Bs_x[i]), len(Bs_y[i])])
        sin_sq = mm - np.trace(Xi.T @ Yi @ Yi.T @ Xi)
        if np.isclose(sin_sq,0):
            sin_sq = 0
        elif sin_sq < 0:
            print('sine squared less than 0')
            print(sin_sq)
            sin_sq = 0
        
        if manifold == 'Grassmann':
            dist += np.sqrt(sin_sq)
        elif manifold == 'Flag':
            dist += sin_sq
        else:
            print('Manifold not recognized')
    
    if manifold == 'Flag':
        dist = np.sqrt(dist)

    return dist

def truncate_svd(C: np.array, n_vecs: int = 0) -> np.array:
    U,_,_ = np.linalg.svd(C, full_matrices=False)

    U = U[:,:n_vecs]

    return U 


def irls_svd(C: np.array, n_vecs: int = 0) -> np.array:
    
    U0 = truncate_svd(C, n_vecs)
    ii=0
    err = 1

    pi = C.shape[1]
    
    while ii < 100 and err > 1e-10:
        C_weighted = []
        weights = np.zeros(pi)
        for i in range(pi):
            c = C[:,[i]]
            sin_sq = np.linalg.norm(c -U0 @ U0.T @ c)
            sin_sq = np.max(np.array([sin_sq, 1e-8]))
            weights[i] = sin_sq**(-1/2)
        
        C_weighted =  C @ np.diag(weights)



        U1 = truncate_svd(C_weighted, n_vecs)
        err = np.abs(np.linalg.norm(U1 @ U1.T - U0 @ U0.T))
        U0 = U1.copy()
        ii+=1


    return U0

def relative_log_mse(D_rec, D_true):
    return 10*np.log10(np.linalg.norm(D_rec - D_true, 'fro')**2/(np.linalg.norm(D_true, 'fro')**2))

def MSE(D_rec, D_true, p):
    return np.sum(np.linalg.norm(D_rec- D_true, axis = 1)**2)/p 

def compute_noise_fraction(noise_exp):
    return .001*noise_exp#10**(-noise_exp)

def compute_outlier_prop(noise_exp):
    return .05*noise_exp

def make_Bs(fl_type):
    Bs = [np.arange(fl_type[0])]
    for i in range(1,len(fl_type)):
        Bs.append(np.arange(fl_type[i-1],fl_type[i]))
    return Bs

def generate_data_noise(noise_fraction, noise_dist, n, col_ids, ms):

    fl_ids = np.cumsum(ms)
    n_k = fl_ids[-1]

    # true flag
    X_true = np.linalg.qr(np.random.normal(size = (n,n_k)))[0][:,:n_k]

    p = np.sum(col_ids)
    
    D_true = np.hstack([X_true[:,:fl_ids[i]] @ np.random.normal(size = (fl_ids[i],col_ids[i])) for i in range(len(col_ids))])
 

    if noise_dist == 'Normal':
        noise = np.random.normal(scale = noise_fraction, size = (n,p))
    elif noise_dist == 'Exponential':
        noise = np.random.exponential(scale = noise_fraction, size = (n,p))
    elif noise_dist == 'Uniform':
        noise = noise_fraction*np.random.uniform(size = (n,p))
    
    D = D_true + noise

    snr = 10*np.log10(np.linalg.trace(D_true@D_true.T)/np.linalg.trace(noise@noise.T))

    return D, D_true, X_true, snr

def generate_data_outliers(prop_outliers, n, col_ids, ms, verbose = 0):

    #flag type with a 0 in front
    fl_ids = np.cumsum(ms)
    n_k = fl_ids[-1]

    # true flag
    X_true = np.linalg.qr(np.random.normal(size = (n,n_k)))[0][:,:n_k]

    p = np.sum(col_ids)
    n_outliers = int(np.floor(prop_outliers*p))
    outlier_subspaces = np.random.randint(len(ms), size = n_outliers)


    D = []
    inlier_ids = []
    outlier_ids = []
    jj=0
    for i in range(len(ms)):
        pi = col_ids[i]
        # get flag and dimension
        Xi = X_true[:,:fl_ids[i]]

        n_outliers_i = len(np.where(outlier_subspaces == i)[0])
        n_inliers_i = pi - n_outliers_i

        Bi = []
        for _ in range(n_inliers_i):
            Bi.append(Xi @ np.random.normal(size = (fl_ids[i],1)))
            inlier_ids.append(jj)
            jj+=1
        for _ in range(n_outliers_i):
            Bi.append((np.eye(n) - X_true @ X_true.T) @ np.random.normal(size = (n,1)))
            outlier_ids.append(jj)
            jj+=1
        
        Bi = np.hstack(Bi)
    
        D.append(Bi)
    
    D = np.hstack(D)

    return D, X_true, np.array(inlier_ids), np.array(outlier_ids)
