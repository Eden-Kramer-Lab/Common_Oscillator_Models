function [x_smth,v_smth] = std_smth(A,Q,x_filter,v_filter)
    
% 2022/12/01 
% Matlab version: R2021b
% This script is for the standard RTS smoother step.

% X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- process state
% y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

% Input:
% x_filter,v_filter are from the filtering step (std_kf.m).

% Output:
% x_smth: estimated hidden signal
% v_smth: estimated covariance

T = size(x_filter,2)-1;
dim_state = size(A,1);         

S = zeros(dim_state,dim_state,T+1);
X_RTS = zeros(dim_state,T+1);
V_RTS = zeros(dim_state,dim_state,T+1);

% set initial backward values
X_RTS(:,T+1) = x_filter(:,T+1);
V_RTS(:,:,T+1) = v_filter(:,:,T+1);

% RTS smoother
for t=T:-1:2
    % time update for each state
    x_minus = A * x_filter(:,t);    
    V_minus = A * v_filter(:,:,t) * A' + Q;  
    
    % smoother gain
    S(:,:,t) = v_filter(:,:,t) * A' * inv(V_minus);
    
    % update
    X_RTS(:,t) = x_filter(:,t) + S(:,:,t)*(X_RTS(:,t+1) - x_minus);
    V_RTS(:,:,t) = v_filter(:,:,t) + S(:,:,t)*(V_RTS(:,:,t+1) - V_minus)*S(:,:,t)';
end
x_smth = X_RTS(:,2:T+1);
v_smth = V_RTS(:,:,2:T+1);
end


  
