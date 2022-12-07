function [W_j,X_j,V_j,KT] = skf(y,A,B,Q,R,X0,Z,pi0)

% 2022/12/01 
% Matlab version: R2021b
% This script is for filtering step.

% Model:
% S_t: Markov chain, P(St=j|St-1) = Zij, transition matrix
% X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- oscillatory latent state
% y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

% Input:
% y: observations
% X_0: initial value of the latent state
% Z: transition matrix associated with the switching state
% pi0: initial prob of the switching state, P(S1).

% Output:
% W_j: state prob. given y, P(St=j|y_1:t)
% X_j: E(Xt|y_1:t, St=j)
% V_j: Cov(Xt|y_1:t, St=j)
% KT: the Kalman gain at time T, which will be used in smoothering

[T, n_obs] = size(y);
dim_state = size(A,1);           % dimention of the state
M = size(Z,1);                   % number of switching states

x_ij = zeros(dim_state,M,M);              % posterior state est obtained by filtering
V_ij = zeros(dim_state,dim_state,M,M,T);  % posterior error covariance matrix est
V_cov_fil = zeros(dim_state,dim_state,M,M,T);

X = zeros(dim_state,M,T+1);               % new state X_j obatained by collapsing x_ij
V = zeros(dim_state,dim_state,M,T+1);     % new cov V_j obatained by collapsing V_ij

% set up initial values
X(:,:,1) = repmat(X0,1,M);      
V_0 = eye(dim_state);        
for i=1:M
    V(:,:,i,1) = V_0;
end
I = eye(dim_state);

W_j = zeros(T+1,M);     % P(S_t=j|y_1:t)
W_j(1,:) = pi0;

% set up variables
y = [zeros(1,n_obs);y];
X_hat = zeros(dim_state,T);
K = zeros(dim_state,n_obs,M,M,T);
W_ij = zeros(M,M,T+1);
L = zeros(M,M);
numr = zeros(M,M,T+1);
W_norm = zeros(T+1,1);

for t=2:T+1
    for j=1:M
        A_j = A(:,:,j);
        B_j = B(:,:,j);
        Q_j = Q(:,:,j);
        R_j = R(:,:,j);

        % kalman filter
        for i=1:M
            % time update for each state
            x_minus = A_j * X(:,i,t-1);                % prior state est
            V_minus = A_j * V(:,:,i,t-1) * A_j' + Q_j; % prior cov. est

            % measurement update
            K(:,:,i,j,t) = (V_minus * B_j') * inv(B_j*V_minus*B_j' + R_j);
            x_ij(:,i,j) = x_minus + K(:,:,i,j,t)*(y(t,:)' - B_j*x_minus);
            V_ij(:,:,i,j,t) = (I - K(:,:,i,j,t)*B_j)*V_minus;

            % one-step covariance
            V_cov_fil(:,:,i,j,t) = (I - K(:,:,i,j,t)*B_j)*A_j* V_ij(:,:,i,j,t-1);

            % likelihood of observing y_t given y_1:t-1, S_t=j, S_t-1=i
            msr_res = y(t,:) - (B_j*x_minus)';
            covar = B_j*V_minus*B_j' + R_j;
            L(i,j) = (det(covar))^(-1/2)*exp((-1/2)*(msr_res*inv(covar))*msr_res');

            % numerator of W_ij
            numr(i,j,t) = L(i,j)*Z(i,j)*W_j(t-1,i);
        end
    end

    % denominator of W_ij
    W_norm(t) = sum(numr(:,:,t),'all');

    % compute W_ij
    W_ij(:,:,t) = numr(:,:,t)/W_norm(t);

    % W_j = P(St=j|y_1:t)
    for j=1:M
        W_j(t,j) = sum(W_ij(:,j,t));       
    end

    % g_ij = P(St-1=i|St=j,y_1:t) = weights of state components
    g = zeros(M,M);
    for j=1:M
        for i=1:M
            g(i,j) = W_ij(i,j,t)/W_j(t,j);
        end
    end

    % approximate (using COLLAPSE - moment matching) new state
    for j=1:M
        X(:,j,t) = x_ij(:,:,j)*g(:,j);
        V(:,:,j,t) = zeros(dim_state);
        for i=1:M
            m = x_ij(:,i,j) - X(:,j,t);
            V(:,:,j,t) = V(:,:,j,t) + g(i,j)*(V_ij(:,:,i,j,t) + m*m');
        end
    end

    % [optional] collape again to get single est of X
    X_hat(:,t) = zeros(dim_state,1);
    for j=1:M
        X_hat(:,t) = X_hat(:,t) + W_j(t,j) * X(:,j,t);
    end
end

W_j = W_j(2:T+1,:);
X_j = X(:,:,2:T+1);
V_j = V(:,:,:,2:T+1);
KT = K(:,:,:,:,T+1);
end
