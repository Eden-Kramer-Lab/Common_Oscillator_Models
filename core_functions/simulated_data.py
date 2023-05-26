import numpy as np
import scipy
from .core import em_B


def simdata_settings():
    # Define the dimensions of the input data
    k = 2  # # of oscillators
    n = 4  # # of electrodes
    M = 3  # # of switching states

    # Other parameters
    fs = 100 # Sampling frequency

    # State parameters
    osc_freqs = np.asarray([7,7]) # Oscillation frequency for each oscillator
    rhos = np.asarray([0.9,0.9]) # Damping parameter for each oscillator
    var_state_nois = np.asarray([1,1]) # State noise for each oscillator

    # Observation parameters
    var_obs_noi = 1

    # Model matrices
    x_dim = k*2
    A, Q = build_AQ(M,fs,osc_freqs,rhos, var_state_nois) # Transition matrix and state noise covariance
    R = build_R(n,M,var_obs_noi) # Observation noise covariance
 
    B = np.zeros((n, x_dim, M)) # Observation matrix
    B[:, :, 0] = [[0.4, 0, 0, 0], [0, 0, 0.4, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    B[:, :, 1] = [[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, -0.25]]
    B[:, :, 2] = [[0.5, 0, 0, 0], [-0.5, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0.4, 0]]

    Z = np.asarray(
        [[0.998, 0.001, 0.001], [0.001, 0.998, 0.001], [0.001, 0.001, 0.998]]
    ) # Discrete state transition matrix
 
    X0 = np.random.multivariate_normal(np.zeros(x_dim), np.eye(x_dim)).T # Initial oscillatory state
    S0 = 0 # Initial discrete state
 
    return fs, k, n, M, osc_freqs, rhos, var_state_nois, var_obs_noi, A, Q, R, B, Z, X0, S0

def build_AQ(M, fs, osc_freqs, rhos, var_state_nois):

    k = len(osc_freqs)
    assert(len(rhos)==k)
    assert(len(var_state_nois)==k)

    x_dim = k*2
    A = np.zeros((x_dim, x_dim, M))
    Q = np.zeros((x_dim, x_dim, M))

    for i in range(M):
        oscmats = []
        varmats = []
        for osc_freq, rho, var_state_noi in zip(osc_freqs,rhos, var_state_nois):
            theta = (2 * np.pi * osc_freq) * (1 / fs)
            oscmat = np.asarray(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            oscmats.append(rho*oscmat)
            varmats.append(var_state_noi*np.eye(2))

        A[:, :, i] = scipy.linalg.block_diag(*oscmats)
        Q[:, :, i] = scipy.linalg.block_diag(*varmats)

    return A, Q


def build_R(n, M, var_obs_noi):
    R = np.zeros((n, n, M))
    for i in range(M):
        R[:, :, i] = var_obs_noi * np.eye(n)
    return R

def simulate(A, B0, Q, R, Z, X_0, S_0, T, s=None):
    rng = np.random.default_rng(14)

    n = R.shape[0] # # of electrodes
    x_dim = A.shape[0] # # of oscillators*2 == k*2 == continuous hidden state dimension
    M = A.shape[-1] # # of switching states

    blnSimS = s is None
    if blnSimS:
        s = np.zeros(T,dtype=int)
        s[0] = S_0

    x = np.zeros([T,x_dim])
    x[0,:] = X_0
    y = np.zeros([T,n])
    y[0,:] = B0[:,:,S_0]@X_0 + rng.multivariate_normal(np.zeros(n),R[:,:,S_0])
    for t in range(1,T):
        if blnSimS:
            s[t] = np.nonzero(rng.multinomial(1,Z[s[t-1],:]))[0][0] # Save the integer in [0,M-1]
        x[t,:] = A[:,:,s[t]]@x[t-1,:] + rng.multivariate_normal(np.zeros(x_dim),Q[:,:,s[t]])
        y[t,:] = B0[:,:,s[t]]@x[t,:] + rng.multivariate_normal(np.zeros(n),R[:,:,s[t]])
    return y, s, x


