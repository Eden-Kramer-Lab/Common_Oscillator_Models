function [mle_B] = kf_em_B(iter,tol,y,X_0,A,B,Q,R)

% 2022/12/01 
% Matlab version: R2021b
% This script is for EM on B matrices *without* switching components.
% It includes filter (std_kf.m) and smoother (std_smth.m) steps, so 
% make sure these two functions can be called. 

% X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- process state
% y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

% Input:
% y: observations
% tol: tolerance 
% iter: number of iteration
% X_0: initial value of the latent state

% Output:
% mle_B: estimated B matrices

dim_state = size(X_0,1);
[n_obs, T] = size(y);

Bj = zeros(n_obs,dim_state,iter);
Bj(:,:,1) = B;

B1 = zeros(n_obs,dim_state,T);
B2 = zeros(dim_state,dim_state,T);

% parameters set up
val = zeros(n_obs,n_obs,T);
lik = zeros(iter,1);

% filter
[x_filter,v_filter] = std_kf(y,A,Bj(:,:,1),Q,R,X_0);
% smoother
[x_smth,v_smth] = std_smth(A,Q,x_filter,v_filter);

for itr=1:iter
    % E-step
    for t=1:T
          val(:,:,t) = (y(:,t)-Bj(:,:,itr)*x_smth(:,t))*(y(:,t)-Bj(:,:,itr)*x_smth(:,t))'+...
              Bj(:,:,itr)*v_smth(:,:,t)*Bj(:,:,itr)';
    end
    lik(itr) = -.5*trace(inv(R)*sum(val,3));

    % M-step
    for t=1:T
        B1(:,:,t) = y(:,t)*x_smth(:,t)';
        B2(:,:,t) = (v_smth(:,:,t)+x_smth(:,t)*x_smth(:,t)');
    end
    Bj(:,:,itr+1) = sum(B1,3)*inv(sum(B2,3));    % analytic sol
    
    if(abs(Bj(:,:,itr+1)-Bj(:,:,itr)) < tol)
        break
    end
    
    [x_filter,v_filter] = std_kf(y,A,Bj(:,:,itr+1),Q,R,X_0);
    [x_smth,v_smth] = std_smth(A,Q,x_filter,v_filter);
    
    fprintf('iter %g Q function %g \n', itr, lik(itr));
end

mle_B = Bj;
end