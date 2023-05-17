import numpy as np
import scipy


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


def create_simulated_data():

    osc_freq = np.asarray([100, 200])
    fs = 100

    import numpy as np

    # EM on B
    k = 2  # # of oscillators
    n = 4  # # of electrodes
    M = 3  # # of switching states

    A = np.zeros((2 * k, 2 * k, M))
    H = np.zeros((n, 2 * k, M))
    Q = np.zeros((2 * k, 2 * k, M))
    R = np.zeros((n, n, M))

    # Set up for A
    rho = 7
    theta1 = (2 * np.pi * osc_freq[0]) * (1 / fs)
    mat1 = np.asarray(
        [[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]]
    )
    var_obs_noi = 1

    # state1
    A[:, :, 0] = scipy.linalg.block_diag(rho * mat1, rho * mat1)
    Q[:, :, 0] = np.eye(2 * k)  # np.cov(noise[:, 0:jumppoint1, 0])
    R[:, :, 0] = var_obs_noi * np.eye(n)

    # state2
    A[:, :, 1] = scipy.linalg.block_diag(rho * mat1, rho * mat1)
    Q[:, :, 1] = np.eye(2 * k)  # np.cov(noise[:, jumppoint1 + 1:jumppoint2, 1])
    R[:, :, 1] = var_obs_noi * np.eye(n)

    # state3
    A[:, :, 2] = scipy.linalg.block_diag(rho * mat1, rho * mat1)
    Q[:, :, 2] = np.eye(2 * k)  # np.cov(noise[:, jumppoint2 + 1:N, 2])
    R[:, :, 2] = var_obs_noi * np.eye(n)

    # B initialization
    B0 = np.zeros((2 * k, 2 * k, M))
    B0[:, :, 0] = [[0.4, 0, 0, 0], [0, 0, 0.4, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    B0[:, :, 1] = [[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, -0.25]]
    B0[:, :, 2] = [[0.5, 0, 0, 0], [-0.5, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0.4, 0]]

    X_0 = np.random.multivariate_normal(np.zeros(2 * k), np.eye(2 * k)).T
    Z = np.asarray(
        [[0.9999, 0.0001, 0.0001], [0.0001, 0.9999, 0.0001], [0.0001, 0.0001, 0.9999]]
    )

    max_iter = 10
    y = scipy.io.loadmat("/Users/edeno/Downloads/y_sim.mat")["y"].T

    # Set tolerance and number of iterations
    tol = 1e-6
    em_B(y, tol, max_iter, A, B0, Q, R, Z, X_0)
