
% 2022/12/01
% Matlab version: R2021b
% This script is for implementing the common oscillator model
% with one alpha oscillaor and two switching states.
% The anesthesia (propofol) example.


%% Load data
load('eeganes07laplac250_detrend_all.mat'); 
addpath(genpath(pwd))

fs = Fs;                    %sampling frequency
dt = 1/fs;                  %sampling interval
N = size(data_det,1);       %number of points in data
n = size(data_det,2);       %number of electrodes
T = N/fs;                   %total time
t = (dt:dt:T);              %time axis


%% Subtract mean
avg = zeros(n,1);
for i=1:n
    avg(i) = mean(data_det(:,i));
end

for i=1:n
    data_det(:,i) = data_det(:,i) - avg(i);
end


%% Bandpass filtering
% bandpass filter the data from 10-12 Hz

out = zeros(size(data_det,1),size(data_det,2));
for i=1:size(data_det,2)
    % bandpass filter
    out(:,i) = bandpass(data_det(:,i),[10 12],fs);  
end
data_det = out;


%% EM on B
% use all the data to estimate the B matrices

% set up
rng(2238)
y = data_det';
T = size(y,2);
fs = 250;       % sampling frequency

% parameters set up
k = 1;          % specify the number of oscillators
M = 2;          % specify the number of switching states
x_dim = 2*k;

% compute var of y on each channel
var_y = zeros(n,1);
for i =1:n
    var_y(i) = cov(data_det(:,i));
end

% parameters set up
A = zeros(2*k,2*k,M);
Q = zeros(2*k,2*k,M);   %sigma matrix
R = zeros(n,n,M);

% A matrix
rho = .99;          % auto-regressive parameter
osc_freq = 11;      % specify the oscillation frequency (alpha)

theta1 = (2*pi*osc_freq(1))*(1/fs);
mat1 = [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)];

% scale of the var
vq = 1;
vr = 60;

% state1
A(:,:,1) = blkdiag(rho*mat1);
Q(:,:,1) = vq*eye(x_dim);
R(:,:,1) = vr*diag(var_y); 

% state2
A(:,:,2) = blkdiag(rho*mat1);
Q(:,:,2) = vq*eye(x_dim);
R(:,:,2) = vr*diag(var_y); 

% initial B
B0 = .1*rand(n,x_dim,M);    % everything is randomly weakly correlated

% initial values
X_0 = mvnrnd(zeros(2*k,1), eye(2*k))'; 
Z = [1-1e-7 1e-7; 1e-7 1-1e-7];

% EM set up
iter = 15;
tol = 1e-5;

% run the EM algorithm
[mle_B,X_RTS,SW,Q_func] = sw_em_B(y',tol,iter,A,B0,Q,R,Z,X_0);


%% Save results
% save('single_rhythm_model','mle_B','SW','X_RTS','Q_func');

