function [x_filter,v_filter] = std_kf(y,A,B,Q,R,X_0)

% 2022/12/01 
% Matlab version: R2021b
% This script is for the standard Kalman filter step.

% X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- process state
% y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

% Input:
% y: observations
% X_0: initial value of the latent state

% Output:
% x_filter: estimated hidden signal
% v_filter: estimated covariance


[n_obs, T] = size(y);
dim_state = size(X_0,1);            

X = zeros(dim_state,T+1);      
V = zeros(dim_state,dim_state,T+1);  
K = zeros(dim_state,n_obs,T);
inn = zeros(n_obs,T);

% set up initial values
X(:,1) = X_0;       
V(:,:,1) = eye(dim_state);            
I = eye(dim_state);   
y(:,T+1) = y(:,T);

% kalman filter
for t=2:T+1
    % time update for each state
    x_minus = A * X(:,t-1);   % prior state est
    V_minus = A * V(:,:,t-1) * A' + Q; % prior cov. est
    % innovation
    inn(:,t) = y(:,t) - B*x_minus;
    % measurement update
    K(:,:,t) = (V_minus * B') * inv(B*V_minus*B' + R);
    X(:,t) = x_minus + K(:,:,t)*(y(:,t) - B*x_minus);
    V(:,:,t) = (I - K(:,:,t)*B)*V_minus;
end
x_filter = X;
v_filter = V;
end

