import numpy as np
from .core import em_B


def skf_input_data():
    # Define the dimensions of the input data
    dim_state = 5  # Dimension of the latent state
    n_obs = 3  # Number of observations
    M = 3  # Number of switching states
    T = 10  # Length of the time series

    # Generate random data for the input variables
    y = np.random.rand(T, n_obs)
    A = np.random.rand(dim_state, dim_state, M)
    B = np.random.rand(n_obs, dim_state, M)
    Q = np.random.rand(dim_state, dim_state, M)
    R = np.random.rand(n_obs, n_obs, M)
    X0 = np.random.rand(dim_state, M)
    Z = np.random.rand(M, M)
    pi0 = np.random.rand(1, M)

    return (y, A, B, Q, R, X0, Z, pi0)


def simulate(A, B0, Q, R, Z, X_0, S_0, T, s=None):
    rng = np.random.default_rng(14)

    n = R.shape[0] # # of electrodes
    K = A.shape[0] # # of oscillators*2 == k*2 == continuous hidden state dimension
    M = A.shape[-1] # # of switching states

    blnSimS = s is None
    if blnSimS:
        s = np.zeros(T,dtype=int)
        s[0] = S_0

    x = np.zeros([T,K])
    x[0,:] = X_0
    y = np.zeros([T,n])
    y[0,:] = B0[:,:,S_0]@X_0 + rng.multivariate_normal(np.zeros(n),R[:,:,S_0])
    for t in range(1,T):
        if blnSimS:
            s[t] = np.nonzero(rng.multinomial(1,Z[s[t-1],:]))[0][0] # Save the integer in [0,M-1]
        x[t,:] = A[:,:,s[t]]@x[t-1,:] + rng.multivariate_normal(np.zeros(K),Q[:,:,s[t]])
        y[t,:] = B0[:,:,s[t]]@x[t,:] + rng.multivariate_normal(np.zeros(n),R[:,:,s[t]])
    return y, s, x


