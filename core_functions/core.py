import numpy as np


def skf(
    y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    X0: np.ndarray,
    Z: np.ndarray,
    pi0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Switching Kalman Filter

    Model:
    S_t: Markov chain, P(St=j|St-1) = Zij, transition matrix
    X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- oscillatory latent state
    y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

    Parameters
    ----------
    y : np.ndarray, shape (n_time, n_obs_dims)
        Observations
    A : np.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
       Continuous oscillatory latent states transition matrix
    B : np.ndarray, shape (n_obs_dims, n_cont_states, n_discrete_states)
        Measurement matrix (map oscillatory latent state to observations)
    Q : np.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Oscillatory state variance (process noise)
    R : np.ndarray, shape (n_obs_dims, n_obs_dims, n_discrete_states)
        Measurement variance
    X0 : np.ndarray, shape (n_cont_states,)
        Initial value of the latent state
    Z : np.ndarray, shape (n_discrete_states, n_discrete_states)
        Discrete state transition matrix
    pi0 : np.ndarray, shape (n_discrete_states,)
        Initial prob of the switching state

    Returns
    -------
    W_j : np.ndarray, shape (n_time, n_discrete_states)
        State prob. given y, P(St=j|y_1:t)
    X_j : np.ndarray, shape (n_cont_states, n_discrete_states, n_time)
        E(Xt|y_1:t, St=j)
    V_j : np.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states, n_time)
        Cov(Xt|y_1:t, St=j)
    KT : np.ndarray, shape (n_cont_states, n_obs_dims, n_discrete_states, n_discrete_states)
        The Kalman gain at time T

    """

    T, n_obs_dims = y.shape
    n_cont_states = A.shape[0]  # dimension of the state
    n_discrete_states = Z.shape[0]  # number of switching states

    x_ij = np.zeros(
        (n_cont_states, n_discrete_states, n_discrete_states)
    )  # posterior state est obtained by filtering
    V_ij = np.zeros(
        (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states, T)
    )  # posterior error covariance matrix est
    V_cov_fil = np.zeros(
        (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states, T)
    )

    X = np.zeros(
        (n_cont_states, n_discrete_states, T + 1)
    )  # new state X_j obatained by collapsing x_ij
    V = np.zeros(
        (n_cont_states, n_cont_states, n_discrete_states, T + 1)
    )  # new cov V_j obatained by collapsing V_ij

    # set up initial values
    X[:, :, 0] = X0

    V_0 = np.eye(n_cont_states)
    V[:, :, :, 0] = V_0[..., np.newaxis]

    I = np.eye(n_cont_states)

    W_j = np.zeros((T + 1, n_discrete_states))  # P(S_t=j|y_1:t)
    W_j[0, :] = pi0

    # set up variables
    y = np.vstack((np.zeros((1, n_obs_dims)), y))
    X_hat = np.zeros((n_cont_states, T))
    K = np.zeros((n_cont_states, n_obs_dims, n_discrete_states, n_discrete_states, T))
    W_ij = np.zeros((n_discrete_states, n_discrete_states, T + 1))
    L = np.zeros((n_discrete_states, n_discrete_states))
    numr = np.zeros((n_discrete_states, n_discrete_states, T + 1))
    W_norm = np.zeros((T + 1, 1))

    for t in range(1, T):
        for j in range(n_discrete_states):
            A_j = A[:, :, j]
            B_j = B[:, :, j]
            Q_j = Q[:, :, j]
            R_j = R[:, :, j]

            # kalman filter
            for i in range(n_discrete_states):
                # time update for each state
                x_minus = A_j @ X[:, i, t - 1]  # prior state est
                V_minus = A_j @ V[:, :, i, t - 1] @ A_j.T + Q_j  # prior cov. est

                # measurement update
                K[:, :, i, j, t] = (V_minus @ B_j.T) @ np.linalg.inv(
                    B_j @ V_minus @ B_j.T + R_j
                )
                x_ij[:, i, j] = x_minus + K[:, :, i, j, t] @ (y[t, :].T - B_j @ x_minus)
                V_ij[:, :, i, j, t] = (I - K[:, :, i, j, t] @ B_j) @ V_minus

                # one-step covariance
                V_cov_fil[:, :, i, j, t] = (
                    (I - K[:, :, i, j, t] @ B_j) @ A_j @ V_ij[:, :, i, j, t - 1]
                )

                # likelihood of observing y_t given y_1:t-1, S_t=j, S_t-1=i
                msr_res = y[t, :] - (B_j @ x_minus).T
                covar = B_j @ V_minus @ B_j.T + R_j
                L[i, j] = (np.linalg.det(covar)) ** (-0.5) * np.exp(
                    (-0.5) * (msr_res @ np.linalg.inv(covar)) @ msr_res.T
                )

                # numerator of W_ij
                numr[i, j, t] = L[i, j] * Z[i, j] * W_j[t - 1, i]

        # denominator of W_ij
        W_norm[t] = np.sum(numr[:, :, t])

        # compute W_ij
        W_ij[:, :, t] = numr[:, :, t] / W_norm[t]

        # W_j = P(St=j|y_1:t)
        for j in range(n_discrete_states):
            W_j[t, j] = np.sum(W_ij[:, j, t])

        # g_ij = P(St-1=i|St=j,y_1:t) = weights of state components
        g = np.zeros((n_discrete_states, n_discrete_states))
        for j in range(n_discrete_states):
            for i in range(n_discrete_states):
                g[i, j] = W_ij[i, j, t] / W_j[t, j]

        # approximate (using COLLAPSE - moment matching) new state
        for j in range(n_discrete_states):
            X[:, j, t] = x_ij[:, :, j] @ g[:, j]
            V[:, :, j, t] = np.zeros((n_cont_states, n_cont_states))
            for i in range(n_discrete_states):
                m = x_ij[:, i, j] - X[:, j, t]
                V[:, :, j, t] = V[:, :, j, t] + g[i, j] * (
                    V_ij[:, :, i, j, t] + m @ m.T
                )

        # [optional] collape again to get single est of X
        X_hat[:, t] = np.zeros((n_cont_states,))
        for j in range(n_discrete_states):
            X_hat[:, t] = X_hat[:, t] + W_j[t, j] * X[:, j, t]

    # Output
    W_j = W_j[1:, :]
    X_j = X[:, :, 1:]
    V_j = V[:, :, :, 1:]
    KT = K[:, :, :, :, T - 1]

    return W_j, X_j, V_j, KT


def smoother(
    y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    X_j: np.ndarray,
    V_j: np.ndarray,
    W_j: np.ndarray,
    KT: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Switching Kalman Smoother

    Model:
    S_t: Markov chain, P(St=j|St-1) = Zij, transition matrix
    X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- oscillatory latent state
    y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

    Parameters
    ----------
    y : np.ndarray, shape (n_time, n_obs_dims)
        Observations
    A : np.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
       Continuous oscillatory latent states transition matrix
    B : np.ndarray, shape (n_obs_dims, n_cont_states, n_discrete_states)
        Measurement matrix (map oscillatory latent state to observations)
    Q : np.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Oscillatory state variance
    R : np.ndarray, shape (n_obs_dims, n_obs_dims, n_discrete_states)
        Measurement variance
    Z : np.ndarray, shape (n_discrete_states, n_discrete_states)
        Discrete state transition matrix
    X_j : np.ndarray, shape (n_cont_states, n_discrete_states, n_time)
        E(Xt|y_1:t, St=j)
    V_j : np.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states, n_time)
        Cov(Xt|y_1:t, St=j)
    W_j : np.ndarray, shape (n_time, n_discrete_states)
        State prob. given y, P(St=j|y_1:t)
    KT : np.ndarray, shape (n_cont_states, n_obs_dims, n_discrete_states, n_discrete_states)
        The Kalman gain at time T, which will be used in smoothering

    Returns
    -------
    M_j : np.ndarray, shape (n_time, n_discrete_states)
        state prob. given y, P(St=j|y_1:T)
    X_RTS : np.ndarray, shape (n_cont_states, n_time)
        E(Xt|y_1:T)
    V_RTS : np.ndarray, shape (n_cont_states, n_cont_states, n_time)
        Cov(Xt|y_1:T)
    V_cov : np.ndarray, shape (n_cont_states, n_cont_states, n_time)
        Cov(Xt,Xt-1|y_1:T), one-step cov

    """

    T = y.shape[0]
    n_cont_states = A.shape[0]  # dimention of the state
    n_discrete_states = Z.shape[0]  # number of switching states

    x_jk = np.zeros((n_cont_states, n_discrete_states, n_discrete_states, T))
    V_jk = np.zeros(
        (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states, T)
    )
    V_jk_cov = np.zeros(
        (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states, T)
    )

    X = np.zeros((n_cont_states, n_discrete_states, T))
    V = np.zeros((n_cont_states, n_cont_states, n_discrete_states, T))
    S = np.zeros(
        (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states, T)
    )

    M_jk = np.zeros((T, n_discrete_states, n_discrete_states))
    M_j = np.zeros((T, n_discrete_states))
    I = np.eye(n_cont_states)

    # set initial backward values
    for i in range(n_discrete_states):
        X[:, i, T - 1] = X_j[:, i, T - 1]
        V[:, :, i, T - 1] = V_j[:, :, i, T - 1]
        M_j[T - 1, i] = W_j[T - 1, i]

        for j in range(n_discrete_states):
            B_j = B[:, :, j]
            A_j = A[:, :, j]
            V_jk_cov[:, :, i, j, T - 1] = (
                (I - KT[:, :, i, j] @ B_j) @ A_j @ V_j[:, :, j, T - 1]
            )

    # # set initial backward values
    # B_j = B[:, :, :, None]
    # A_j = A[:, :, :, None]
    # V_jk_cov[:, :, :, :, T - 1] = (
    #     (I[:, :, None, None] - KT @ B_j) @ A_j @ V_j[:, :, :, T - 1, None]
    # )
    # X[:, :, T - 1] = X_j[:, :, T - 1]
    # V[:, :, :, T - 1] = V_j[:, :, :, T - 1]
    # M_j[T - 1, :] = W_j[T - 1, :]

    # set up variables
    X_RTS = np.zeros((n_cont_states, T))
    V_RTS = np.zeros((n_cont_states, n_cont_states, T))
    V_cov = np.zeros((n_cont_states, n_cont_states, T))
    U_jk = np.zeros((n_discrete_states, n_discrete_states, T))

    for t in range(T - 2, -1, -1):
        U_norm = np.zeros((n_discrete_states, 1))
        for j in range(n_discrete_states):
            A_j = A[:, :, j]
            Q_j = Q[:, :, j]

            # RTS smoother
            for k in range(n_discrete_states):
                # time update for each state
                x_minus = A_j @ X_j[:, j, t]  # prior state est
                V_minus = A_j @ V_j[:, :, j, t] @ A_j.T + Q_j  # prior cov. est

                # smoother gain
                S[:, :, j, k, t] = V_j[:, :, j, t] @ A_j.T @ np.linalg.inv(V_minus)

                # update
                x_jk[:, j, k, t] = X_j[:, j, t] + S[:, :, j, k, t] @ (
                    X[:, k, t + 1] - x_minus
                )
                V_jk[:, :, j, k, t] = (
                    V_j[:, :, j, t]
                    + S[:, :, j, k, t]
                    @ (V[:, :, k, t + 1] - V_minus)
                    @ S[:, :, j, k, t].T
                )

                # state probability
                U_jk[j, k, t] = (
                    W_j[t, j]
                    * np.linalg.det(V_minus) ** (-0.5)
                    * np.exp(
                        -0.5
                        * (X[:, k, t + 1] - x_minus).T
                        @ np.linalg.inv(V_minus)
                        @ (X[:, k, t + 1] - x_minus)
                    )
                )
                U_norm[j] += U_jk[j, k, t]

            # # RTS smoother
            # S[:, :, :, :, t] = V_j[:, :, :, t] @ A.transpose(0, 2, 1) @ np.linalg.inv(A @ V_j[:, :, :, t] @ A.transpose(0, 2, 1) + Q)

            # # update
            # x_jk[:, :, :, t] = X_j[:, :, t] + S[:, :, :, :, t] @ (X[:, :, t + 1] - A @ X_j[:, :, t])
            # V_jk[:, :, :, :, t] = V_j[:, :, :, t] + S[:, :, :, :, t] @ (V[:, :, :, t + 1] - A @ V_j[:, :, :, t] @ A.transpose(0, 2, 1) - Q) @ S[:, :, :, :, t].transpose(0, 2, 1)

            # # state probability
            # U_jk[:, :, t] = W_j[t, :, None] * np.linalg.det(A @ V_j[:, :, :, t] @ A.transpose(0, 2, 1) + Q) ** (-0.5) * np.exp(-0.5 * np.sum((X[:, :, t + 1] - A @ X_j[:, :, t]) @ np.linalg.inv(A @ V_j[:, :, :, t] @ A.transpose(0, 2, 1) + Q) * (X[:, :, t + 1] - A @ X_j[:, :, t]), axis=0))
            # U_norm[:] = np.sum(U_jk[:, :, t], axis=1)

        for j in range(n_discrete_states):
            for k in range(n_discrete_states):
                M_jk[t, j, k] = U_jk[j, k, t] / U_norm[j]

        # M_jk[t, j, :] = U_jk[j, :, t] / U_norm[j]

        for j in range(n_discrete_states):
            X_RTS[:, t] += X[:, j, t] * M_j[t, j]
            V_RTS[:, :, t] += V[:, :, j, t] * M_j[t, j]
            V_cov[:, :, t] += V_jk_cov[:, :, j, j, t] * M_j[t, j]

            for k in range(n_discrete_states):
                V_RTS[:, :, t] += (
                    (x_jk[:, j, k, t] - X_RTS[:, t])
                    @ (x_jk[:, j, k, t] - X_RTS[:, t]).T
                    * M_jk[t, j, k]
                )

        # X_RTS[:, t] = X[:, :, t] @ M_j[t, :].T
        # V_RTS[:, :, t] = V[:, :, :, t] @ M_j[t, :].T + np.einsum("jk,j,k->ij", x_jk[:, :, :, t] - X_RTS[:, t], M_j[t, :], M_j[t, :])
        # V_cov[:, :, t] = V_jk_cov[:, :, :, :, t] @ M_j[t, :].T

    return M_j, X_RTS, V_RTS, V_cov


def em_B(
    y: np.ndarray,
    tol: float,
    max_iter: int,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    X_0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EM

    Model:
    S_t: Markov chain, P(St=j|St-1) = Zij, transition matrix
    X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- oscillatory latent state
    y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

    Parameters
    ----------
    y : np.ndarray, shape (n_time, n_obs_dims)
        Observations
    tol : float
    max_iter : int
    A : np.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
       Continuous oscillatory latent states transition matrix
    B : np.ndarray, shape (n_obs_dims, n_cont_states, n_discrete_states)
        Measurement matrix (map oscillatory latent state to observations)
    Q : np.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Oscillatory state variance
    R : np.ndarray, shape (n_obs_dims, n_obs_dims, n_discrete_states)
        Measurement variance
    Z : np.ndarray, shape (n_discrete_states, n_discrete_states)
        Discrete state transition matrix
    X_0 : np.ndarray, shape (n_cont_states,)
        Initial value of the latent state


    Returns
    -------
    mle_B : np.ndarray, shape (n_observation_dims, n_cont_states, n_discrete_states, max_iter-1)
        estimated B matrices
    X_RTS : np.ndarray, shape (n_cont_states, n_time)
        E(Xt|y_1:T)
    SW : np.ndarray, shape (n_time, n_discrete_states, max_iter+1)
    Q_func : np.ndarray, shape (max_iter,)

    """

    n_discrete_states = Z.shape[0]
    n_cont_states = A.shape[0]
    T, n_obs_dims = y.shape

    pi0 = (
        np.ones(n_discrete_states) / n_discrete_states
    )  # initial prob of the switching state

    # parameters set up
    Bj = np.zeros((n_obs_dims, n_cont_states, n_discrete_states, max_iter))
    # set up initial B
    Bj[:, :, :, 0] = B

    B1 = np.zeros((n_obs_dims, n_cont_states, n_discrete_states, T))
    B2 = np.zeros((n_cont_states, n_cont_states, n_discrete_states, T))
    B1sum = np.zeros((n_obs_dims, n_cont_states, T))
    B2sum = np.zeros((n_cont_states, n_cont_states, T))

    ins = np.zeros((T, n_discrete_states))
    Q_func = np.zeros(max_iter)
    SW = np.zeros((T, n_discrete_states, max_iter + 1))

    # filter
    W_j, X_j, V_j, KT = skf(y, A, Bj[:, :, :, 0], Q, R, X_0, Z, pi0)
    # smoother
    M_j, X_RTS, V_RTS, V_cov = smoother(
        y, A, Bj[:, :, :, 0], Q, R, Z, X_j, V_j, W_j, KT
    )

    SW[:, :, 0] = M_j

    for itr in range(max_iter):
        # E-step
        for j in range(n_discrete_states):
            A_j = A[:, :, j]
            Q_j = Q[:, :, j]
            R_j = R[:, :, j]
            B_j = Bj[:, :, j, itr]

            for t in range(1, T):
                ins[t, j] = M_j[t, j] * (
                    np.log(np.linalg.det(Q_j))
                    + np.log(np.linalg.det(R_j))
                    + np.trace(
                        np.linalg.inv(R_j)
                        * (
                            y[t, :].T @ y[t, :]
                            + B_j
                            @ (V_RTS[:, :, t] + X_RTS[:, t] @ X_RTS[:, t].T)
                            @ B_j.T
                            - B_j @ X_RTS[:, t] @ y[t, :]
                            - y[t, :].T @ X_RTS[:, t].T @ B_j.T
                        )
                    )
                    + np.trace(
                        np.linalg.inv(Q_j)
                        * (
                            V_RTS[:, :, t]
                            + X_RTS[:, t] @ X_RTS[:, t].T
                            - A_j @ (V_cov[:, :, t] + X_RTS[:, t] @ X_RTS[:, t - 1].T).T
                            - (V_cov[:, :, t] + X_RTS[:, t] @ X_RTS[:, t - 1].T) @ A_j.T
                            + A_j
                            @ (V_RTS[:, :, t - 1] + X_RTS[:, t - 1] @ X_RTS[:, t - 1].T)
                            @ A_j.T
                        )
                    )
                )
        lik = -0.5 * np.sum(ins)

        # M-step
        for j in range(n_discrete_states):
            for t in range(1, T):
                B1[:, :, j, t] = M_j[t, j] * (y[t, :].T @ X_RTS[:, t])
                B2[:, :, j, t] = M_j[t, j] * (
                    V_RTS[:, :, t] + X_RTS[:, t] @ X_RTS[:, t].T
                )

            B1sum[:, :, j] = np.sum(B1[:, :, j, :], axis=3)
            B2sum[:, :, j] = np.sum(B2[:, :, j, :], axis=3)

            Bj[:, :, j, itr + 1] = (
                np.linalg.inv(B2sum[:, :, j]) @ B1sum[:, :, j]
            )  # analytic sol

        if abs(Bj[:, :, 1, itr + 1] - Bj[:, :, 1, itr]) < tol:
            break

        W_j, X_j, V_j, KT = skf(y, A, Bj[:, :, :, itr + 1], Q, R, X_0, Z, pi0)
        M_j, X_RTS, V_RTS, V_cov = smoother(
            y, A, Bj[:, :, :, itr + 1], Q, R, Z, X_j, V_j, W_j, KT
        )

        Q_func[itr] = lik
        print("iter: {}, Q-function: {}".format(itr, lik))

        SW[:, :, itr + 1] = M_j

    mle_B = Bj[:, :, :, 1:]

    return mle_B, X_RTS, SW, Q_func

import numpy as np

def get_theoretical_psd(f_y, Fs, freq_tot, ampl_tot_k, nois_tot_k):
    """
    Returns theoretical/parametric power spectral density (PSD) for a signal generated with Matsuda's et al. (2017) model.
    Based on Hugo Soulat's https://github.com/mh105/SSP/ssp_decomp/get_theoretical_psd.m

    Args:
        f_y (array-like): Frequencies at which PSD is calculated.
        Fs (float): Sampling frequency.
        freq_tot (array-like): Peak frequencies of the oscillators.
        ampl_tot_k (array-like): Amplitudes of the oscillators.
        nois_tot_k (array-like): State noise covariance.

    Returns:
        H_tot (array): Total PSD.
        H_i (array): PSD of a given oscillation at each frequency in freq_tot.

    """

    Nfreq = len(freq_tot)

    # Solve small issue when f_i = Fs / 4 -> w = pi/2
    for ww_i in range(len(freq_tot)):
        if freq_tot[ww_i] == Fs / 4:
            freq_tot[ww_i] = freq_tot[ww_i] + freq_tot[ww_i] * 0.002

    z = np.exp(2j * np.pi * f_y / Fs)
    w_tot = freq_tot * 2 * np.pi / Fs

    nois2 = nois_tot_k

    A_i = (1 - 2 * ampl_tot_k ** 2 * np.cos(w_tot) ** 2 + ampl_tot_k ** 4 * np.cos(2 * w_tot)) / (
            ampl_tot_k * (ampl_tot_k ** 2 - 1) * np.cos(w_tot))
    B_i = 0.5 * (A_i - 2 * ampl_tot_k * np.cos(w_tot) + np.sqrt((A_i - 2 * ampl_tot_k * np.cos(w_tot)) ** 2 - 4))
    V_i = -(nois2 * ampl_tot_k * np.cos(w_tot)) / B_i

    H_i = np.zeros((Nfreq, len(z)))
    for ii in range(Nfreq):
        H_i[ii, :] = (V_i[ii] / Fs) * np.abs(1 + B_i[ii] * z) ** 2 / np.abs(
            1 - 2 * ampl_tot_k[ii] * np.cos(w_tot[ii]) * z + ampl_tot_k[ii] ** 2 * z ** 2) ** 2

    H_tot = np.sum(H_i, axis=0)

    return H_tot, H_i
