function [mle_B,X_RTS,SW,Q_func] = em_B(y,tol,iter,A,B,Q,R,Z,X_0)

% 2022/12/01 
% Matlab version: R2021b
% This script is for EM on B matrices with switching components.
% It includes filter (skf.m) and smoother (smoother.m) steps, so 
% make sure these two functions can be called. 

% Model:
% S_t: Markov chain, P(St=j|St-1) = Zij, transition matrix
% X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- oscillatory latent state
% y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

% Input:
% y: observations
% tol: tolerance 
% iter: number of iteration
% X_0: initial value of the latent state

% Output:
% mle_B: estimated B matrices
% X_RTS: E(Xt|y_1:T)
% V_RTS: Cov(Xt|y_1:T)
% V_cov: Cov(Xt,Xt-1|y_1:T), one-step cov

M = size(Z,1);
dim_state = size(A,1);
[T, n_obs] = size(y);

pi0 = ones(1,M)/M;      % initial prob of the switching state

% parameters set up
Bj = zeros(n_obs,dim_state,M,iter);
% set up initial B
for i=1:M
    Bj(:,:,i,1) = B(:,:,i);
end

B1 = zeros(n_obs,dim_state,M,T);
B2 = zeros(dim_state,dim_state,M,T);
B1sum = zeros(n_obs,dim_state,T);
B2sum = zeros(dim_state,dim_state,T);

ins = zeros(T,M);
Q_func = zeros(iter,1);
SW = zeros(T,M,iter+1);

% filter
[W_j,X_j,V_j,KT] = skf(y,A,Bj(:,:,:,1),Q,R,X_0,Z,pi0);
% smoother
[M_j,X_RTS,V_RTS,V_cov] = smoother(y,A,Bj(:,:,:,1),Q,R,Z,X_j,V_j,W_j,KT);

SW(:,:,1) = M_j;

for itr=1:iter
    % E-step 
    for j=1:M
        A_j = A(:,:,j);
        Q_j = Q(:,:,j);
        R_j = R(:,:,j);
        B_j = Bj(:,:,j,itr);
        
        for t=2:T
            ins(t,j) = M_j(t,j) * (log(det(Q_j)) + log(det(R_j)) + ...
                trace(inv(R_j)*(y(t,:)'*y(t,:) + ...
                    B_j*(V_RTS(:,:,t) + X_RTS(:,t)*X_RTS(:,t)')*B_j' - ...
                    B_j*X_RTS(:,t)*y(t,:) - y(t,:)'*X_RTS(:,t)'*B_j')) + ...
                trace(inv(Q_j)*(V_RTS(:,:,t)+X_RTS(:,t)*X_RTS(:,t)' - ...
                    A_j*(V_cov(:,:,t)+X_RTS(:,t)*X_RTS(:,t-1)')' - ...
                    (V_cov(:,:,t)+X_RTS(:,t)*X_RTS(:,t-1)')*A_j' + ...
                    A_j*(V_RTS(:,:,t-1) + X_RTS(:,t-1)*X_RTS(:,t-1)')*A_j')));
        end
    end
    lik = -0.5*sum(ins,'all');            
    
    % M-step
    for j=1:M
        for t=2:T
            B1(:,:,j,t) = M_j(t,j) * (y(t,:)' * X_RTS(:,t)');
            B2(:,:,j,t) = M_j(t,j) * (V_RTS(:,:,t) + X_RTS(:,t)*X_RTS(:,t)');
        end
        B1sum(:,:,j) = sum(B1(:,:,j,:),4);
        B2sum(:,:,j) = sum(B2(:,:,j,:),4);
        
        Bj(:,:,j,itr+1) = B1sum(:,:,j)*inv(B2sum(:,:,j));   % analytic sol
    end
    
    if(abs(Bj(:,:,1,itr+1)-Bj(:,:,1,itr)) < tol)
        break
    end
    
    [W_j,X_j,V_j,KT] = skf(y,A,Bj(:,:,:,itr+1),Q,R,X_0,Z,pi0);
    [M_j,X_RTS,V_RTS,V_cov] = smoother(y,A,Bj(:,:,:,itr+1),Q,R,Z,X_j,V_j,W_j,KT);
    
    Q_func(itr) = lik;
    fprintf('iter %g Q function %g \n', itr, lik);
    
    SW(:,:,itr+1) = M_j;
end

mle_B = Bj(:,:,:,2:itr+1);

end