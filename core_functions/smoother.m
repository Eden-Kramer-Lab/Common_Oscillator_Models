function [M_j,X_RTS,V_RTS,V_cov] = smoother(y,A,B,Q,R,Z,X_j,V_j,W_j,KT)

% 2022/12/01 
% Matlab version: R2021b
% This script is for smoothering step.

% Model:
% S_t: Markov chain, P(St=j|St-1) = Zij, transition matrix
% X_t = A * X_t-1 + u_t  , u_t ~ N(0, Q)  -- oscillatory latent state
% y_t = B * X_t + v_t    , v_t ~ N(0, R)  -- observation

% Input:
% y: observations
% Z: transition matrix associated with the switching state
% X_j,V_j,W_j,KT are from the filtering step
    
% Output:
% M_j: state prob. given y, P(St=j|y_1:T)
% X_RTS: E(Xt|y_1:T)
% V_RTS: Cov(Xt|y_1:T)
% V_cov: Cov(Xt,Xt-1|y_1:T), one-step cov
    
T = size(y,1);
dim_state = size(A,1);           % dimention of the state
M = size(Z,1);                   % number of switching states

x_jk = zeros(dim_state,M,M,T);     
V_jk = zeros(dim_state,dim_state,M,M,T);
V_jk_cov = zeros(dim_state,dim_state,M,M,T);

X = zeros(dim_state,M,T);      
V = zeros(dim_state,dim_state,M,T);
S = zeros(dim_state,dim_state,M,M,T);

M_jk = zeros(T,M,M);
M_j = zeros(T,M); 
I = eye(dim_state);

% set initial backward values
for i=1:M
    X(:,i,T) = X_j(:,i,T);
    V(:,:,i,T) = V_j(:,:,i,T);
    M_j(T,i) = W_j(T,i);
    
    for j=1:M
        B_j = B(:,:,j);
        A_j = A(:,:,j);
        V_jk_cov(:,:,i,j,T) = (I - KT(:,:,i,j)*B_j) * A_j * V_j(:,:,j,T);
    end
end

% set up variables
X_RTS = zeros(dim_state,T);
V_RTS = zeros(dim_state,dim_state,T);
U_jk = zeros(M,M,T);

for t=(T-1):-1:1
  U_norm = zeros(M,1);
  for j=1:M
    A_j = A(:,:,j);
    Q_j = Q(:,:,j);
    
    % RTS smoother
    for k=1:M
      % time update for each state
      x_minus = A_j * X_j(:,j,t);   % prior state est 
      V_minus = A_j * V_j(:,:,j,t) * A_j' + Q_j; % prior cov. est 
      
      % smoother gain
      S(:,:,j,k,t) = V_j(:,:,j,t) * A_j' * inv(V_minus);
      
      % update
      x_jk(:,j,k,t) = X_j(:,j,t) + S(:,:,j,k,t)*(X(:,k,t+1) - x_minus);
      V_jk(:,:,j,k,t) = V_j(:,:,j,t) + S(:,:,j,k,t)*(V(:,:,k,t+1) - V_minus)*S(:,:,j,k,t)';
      U_jk(j,k,t) = W_j(t,j) * Z(j,k); % numerator of U_jk
    end 
  end
 
  for k=1:M
      U_norm(k) = sum(U_jk(:,k,t));
  end

  for j=1:M
      for k=1:M
        U_jk(j,k,t) = U_jk(j,k,t)/U_norm(k); 
        M_jk(t,j,k) = U_jk(j,k,t) * M_j(t+1,k);     % M_jk
      end
  end

  % M_j = P(St=j|y_1:T)
  for j=1:M
    M_j(t,j) = sum(M_jk(t,j,:));                    % M_j = P(S_t|y=1:T)
  end

  % weights of state components
  g = zeros(M,M);
  for k=1:M
    for j=1:M
        g(j,k) = M_jk(t,j,k)/M_j(t,j); 
    end
  end

  % approximate (using COLLAPSE) new state
  for j=1:M
    X(:,j,t) = squeeze(x_jk(:,j,:,t)) * g(j,:)'; % X_t^j
    V(:,:,j,t) = zeros(dim_state);               % V_t^j: (dim(X)*dim(X)*M*t)
    for k=1:M
      d = x_jk(:,j,k,t) - X(:,j,t);
      V(:,:,j,t) = V(:,:,j,t) + g(j,k)*(V_jk(:,:,j,k,t) + d*d'); 
    end
  end
  
  % collapse again to get a single estimate
  for j=1:M
      X_RTS(:,t) = X_RTS(:,t) + M_j(t,j) * X(:,j,t); % X_t hat
  end
  for j=1:M
      diff = X(:,j,t) - X_RTS(:,t);
      V_RTS(:,:,t) = V_RTS(:,:,t) + M_j(t,j) * (V(:,:,j,t) + diff*diff'); % V_t hat
  end 
end


% one-step convariance
% ====================
V_k_cov = zeros(dim_state,dim_state,M,T);
B1 = zeros(dim_state,M,T);
B2 = zeros(dim_state,M,T);

V_cov = zeros(dim_state,dim_state,T);
B3 = zeros(dim_state,T);
B4 = zeros(dim_state,T);

% V_jk_cov = P(Xt,Xt+1|y_1:T,St=j,St+1=k)
for t=(T-1):-1:2
    for j=1:M
        A_j = A(:,:,j);
        for k=1:M
            V_jk_cov(:,:,j,k,t) = V_j(:,:,j,t) * S(:,:,j,k,t-1)' + ...
                S(:,:,j,k,t)*(V_jk_cov(:,:,j,k,t+1) - A_j*V_j(:,:,j,t))*S(:,:,j,k,t-1)';
        end
    end

    % collapsing for V_k_cov
    for k=1:M
        B1(:,k,t) = x_jk(:,:,k,t+1) * U_jk(:,k,t);
        B2(:,k,t) = x_jk(:,:,k,t) * U_jk(:,k,t);
        for j=1:M
            d1 = x_jk(:,j,k,t+1)-B1(:,k,t);
            d2 = x_jk(:,j,k,t)-B2(:,k,t);
            V_k_cov(:,:,k,t) = V_k_cov(:,:,k,t) + ...       % V_k_cov
                U_jk(j,k,t)*(V_jk_cov(:,:,j,k,t) + d1*d2');
        end
    end
    
    % collapsing again for V_cov = P(Xt,Xt+1|y_1:T)
    for k=1:M
        B3(:,t) = B3(:,t) + M_j(t,k) * X(:,k,t+1);
        B4(:,t) = B4(:,t) + M_j(t,k) * X(:,k,t);
    end
    
    for k=1:M
        diff1 = X(:,k,t+1)-B3(:,t);
        diff2 = X(:,k,t)-B4(:,t);
        V_cov(:,:,t) = V_cov(:,:,t) + ...
            M_j(t+1,k)*(V_k_cov(:,:,k,t) + diff1*diff2');
    end
end
