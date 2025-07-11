import numpy as np
import scipy.linalg as La
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0,
    os.path.join(os.path.dirname(__file__), '../'))

from src.data_generation import simulate_Covariance, phasegeneration, \
    simulate_gaussiandata, simulate_scaledgaussiandata
from src.covariance_estimators import SCM, regul_linear, bandw, SCM_LR, \
    tyler_estimator_covariance, tyler_estimator_covariance_LR, corr_phase, corr_phase_LR
from src.optimization import MM_KL_IPL, MM_LS_IPL, RG_comet_IPL, RG_LS_IPL

def data_gen(size=1000, p=30, n=64, cost='LS', estimator='PO', regul='SK'):
    rho_list = np.random.uniform(low=0.96, high=0.99, size=200)
    b = 3 # bandwidth parameter
    alpha = 0.5 # coefficient regularization
    rank = 1 # rank of the covariance matrix (p if full-rank)
    phasechoice = 'random'
    nu = 0

    delta_thetasim_list = []
    SigmaTrue_list = []
    trueCov_list = []
    X_list = []
    Sigma_tilde_list = [] 
    w_theta_list = []

    for rho in rho_list:
        #Génération des matrices de covariance estimées sigma_tilde
        delta_thetasim = phasegeneration(phasechoice,p) #generate phase with either random or linear. for linear, define last phase is needed
        SigmaTrue = La.toeplitz(rho**np.arange(p))
        trueCov = simulate_Covariance(SigmaTrue, delta_thetasim)

        delta_thetasim_list.append(delta_thetasim)
        SigmaTrue_list.append(SigmaTrue)
        trueCov_list.append(trueCov)

        if nu == 0: 
            X = simulate_gaussiandata(trueCov, p,n)
        else:
            X = simulate_scaledgaussiandata(trueCov, nu, p,n)

        X_list.append(X)

        if estimator == 'SCM':
            Sigma = SCM(X)
            if regul == 'LR':
                Sigma_tilde = SCM_LR(X,rank)
            if regul == 'SK': 
                Sigma_tilde = regul_linear(Sigma,alpha)
            if regul == 'BW':
                Sigma_tilde = bandw(Sigma,b)
            if regul == False:
                Sigma_tilde = Sigma

        if estimator == 'PO':
            Sigma = corr_phase(X)
            if regul == 'LR':
                Sigma_tilde = corr_phase_LR(X,rank)
            if regul == 'SK': 
                Sigma_tilde = regul_linear(Sigma,alpha)
            if regul == 'BW':
                Sigma_tilde = bandw(Sigma,b)
            if regul == False:
                Sigma_tilde = Sigma
        Sigma_tilde_list.append(Sigma_tilde)
        w_theta = RG_LS_IPL(Sigma_tilde, 100, True, False, False)
        w_theta_list.append(w_theta)
    
    return X_list, w_theta_list, delta_thetasim_list, rho_list


def normalize_data(X, w_list):
    real_parts = np.array([np.real(x) for x in X])
    imag_parts = np.array([np.imag(x) for x in X])
    # Normalize real and imaginary parts separately
    real_mean, real_std = np.mean(real_parts), np.std(real_parts)
    imag_mean, imag_std = np.mean(imag_parts), np.std(imag_parts)
    normalized_real = (real_parts - real_mean) / real_std
    normalized_imag = (imag_parts - imag_mean) / imag_std
    # Reconstruct complex values
    X = normalized_real + 1j * normalized_imag
    # Normalize w_theta_list
    original_norms = [np.linalg.norm(w) for w in w_list]
    w_list = [w / np.linalg.norm(w) if np.linalg.norm(w) != 0 else w for w in w_list]

    return X, w_list, real_mean, real_std, imag_mean, imag_std, original_norms

def denormalize_data(X, real_mean, real_std, imag_mean, imag_std):
    real_parts = np.real(X)
    imag_parts = np.imag(X)

    # Dénormaliser les parties réelles et imaginaires
    denormalized_real = real_parts * real_std + real_mean
    denormalized_imag = imag_parts * imag_std + imag_mean

    # Reconstruire les valeurs complexes
    X = denormalized_real + 1j * denormalized_imag

    return X

def denormalize_w_list(w_list, original_norms):
    denormalized_w_list = [w * original_norms[i] for i, w in enumerate(w_list)]
    return denormalized_w_list