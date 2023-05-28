
% 2023/05/16
% Matlab version: R2021b
% This script is for implementing the common oscillator model
% with two 7 Hz oscillators and three switching states.
% Simulated data

clear variables
close all 

%% Load data
basepath = '..';
addpath(genpath(basepath))

load([basepath '/simdata/y_sim.mat']);

%% set up
rng(2238)
n = size(y,1);
T = size(y,2);
fs = 100;       % sampling frequency

%% EM on B
% use all the data to estimate the B matrices


% parameters set up
k = 2;          % specify the number of oscillators
M = 3;          % specify the number of switching states
x_dim = 2*k;

% parameters set up
A = zeros(2*k,2*k,M);
Q = zeros(2*k,2*k,M);   %sigma matrix
R = zeros(n,n,M);

% A matrix
rho = .9;          % auto-regressive parameter
osc_freq = 7;      % specify the oscillation frequency

% scale of the var
vq = 1;
vr = 1;

theta1 = (2*pi*osc_freq(1))*(1/fs);
mat1 = [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)];


for i=1:M
    A(:,:,i) = blkdiag(rho*mat1, rho*mat1);
    Q(:,:,i) = vq*eye(x_dim);
    R(:,:,i) = vr*eye(n); 
end

ta = (1:T)/fs;              %time axis
Strue = ones(T,1);
Strue(ta>80)=2;
Strue(ta>200)=3;

Btrue = zeros(n, x_dim, M);
Btrue(:, :, 1) = [0.4, 0, 0, 0; 0, 0, 0.4, 0; 0, 0, 0, 0; 0, 0, 0, 0];
Btrue(:, :, 2) = [0.3, 0, 0, 0; 0, 0.3, 0, 0; 0, 0, 0.25, 0; 0, 0, 0, -0.25];
Btrue(:, :, 3) = [0.5, 0, 0, 0; -0.5, 0, 0, 0; 0.5, 0, 0, 0; 0, 0, 0.4, 0];

% (Testing)
% B0 = Btrue;
% X_0 = 0.1*ones(2*k,1); 

% initial values
B0 = .1*rand(n,x_dim,M);    % everything is randomly weakly correlated
X_0 = mvnrnd(zeros(2*k,1), eye(2*k))'; 
Z = 1e-3*ones(M)+(1-3e-3)*eye(M);

% EM set up
iter = 10;
tol = 1e-5;

% run the EM algorithm
[mle_B,X_RTS,SW,Q_func] = em_B(y',tol,iter,A,B0,Q,R,Z,X_0);


%% Save results
% save([basepath '/output/simdata_model','mle_B','SW','X_RTS','Q_func']);

