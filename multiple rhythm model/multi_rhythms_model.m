
% 2022/12/01
% Matlab version: R2021b
% This script is for implementing the common oscillator model
% with one alpha oscillaor and two switching states.
% The anesthesia (propofol) example.
% Note: for estimating the B matrices separately from awake &
% unconscious periods, we use the standard KF and smoother
% (std_kf.m & std_smth.m).For estimating the prob. of the switching 
% states, we use the switching KF and smoother (skf.m & smoother.m).


%% Load data
load('eeganes07laplac250_detrend_all.mat');
addpath(genpath(pwd))

fs = 250;                   %sampling frequency
dt =1/fs;                   %sampling interval
N = size(data_det,1);       %number of points in data
n = size(data_det,2);       %number of electrodes
T = N/fs;                   %total time
t = (dt:dt:TT);             %time axis


%% Subtract mean
avg = zeros(n,1);
for i=1:n
    avg(i) = mean(data_det(:,i));
end

for i=1:n
    data_det(:,i) = data_det(:,i) - avg(i);
end


%% Remove electrodes due to artifacts

% identify the electrode # to be removed
remove = [4,8,9,10,16,35];

% remove electrodes from the dataset
data_det(:,remove) = [];

%% EM on B from awake & unconscious periods separately
% setup
rng(2238)

% ===== awake period =====
T1 = 1600; T2 = 1780;
y = data_det(T1*fs:T2*fs,:)';
n = size(y,1);

% parameters set up
k = 4;          % specify the number of oscillators
fs = 250;
x_dim = 2*k;

% compute var of y on each channel
var_y = zeros(n,1);
for i =1:n
    var_y(i) = cov(data_det(T1*fs:T2*fs,i));
end

% A matrix
rho = .99;          % auto-regressive parameter
fq = [.5,11];       % specify the oscillation frequency (slow + alpha)
osc_freq = fq;

theta1 = (2*pi*osc_freq(1))*(1/fs);
mat1 = [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)];
theta2 = (2*pi*osc_freq(2))*(1/fs);
mat2 = [cos(theta2) -sin(theta2); sin(theta2) cos(theta2)];

% one alpha + three slow waves
A = blkdiag(rho*mat2, rho*mat1, rho*mat1, rho*mat1);  

% Q matrix (sigma)
vq = 1;
Q = vq*eye(x_dim);

% R matrix
vr = 20;
R = vr*diag(var_y);

% initial B
B0 = .1*rand(n,x_dim);

% initial values
X_0 = mvnrnd(zeros(2*k,1), eye(2*k))'; 

% EM set-up
iter = 10;
tol = 1e-5;

% run the EM algorithm
[mle_B_awake] = kf_em_B(iter,tol,y,X_0,A,B0,Q,R);

% ===== unconscious period =====
T1 = 4600; T2 = 4780;
y = data_det(T1*fs:T2*fs,:)';

% compute var of y on each channel
var_y = zeros(n,1);
for i =1:n
    var_y(i) = cov(data_det(T1*fs:T2*fs,i));
end

% initial B
B0 = .1*rand(n,x_dim);

% run the EM algorithm
[mle_B_unc] = kf_em_B(iter,tol,y,X_0,A,B0,Q,R);


%% Save results

% awake
% save('mle_B_awake','mle_B');
% load('mle_B_awake','mle_B');
iter = size(mle_B,3);
B_awake = mle_B(:,:,iter);

% unconscious
% save('mle_B_unc','mle_B');
% load('mle_B_unc','mle_B');
iter = size(mle_B,3);
B_unc = mle_B(:,:,iter);

%% Plot estimated B matrices

% use prp_plot_B_nosw_multi.m

%% Remove time periods from the dataset due to artifacts

% remove the time period at the very end
TT = 8500;
data_det = data_det(1:TT*fs,:);

% identify the time segments to be removed
p = [1900,2030,2795,2900,4085,6050,6265,6908];
range = [20,20,40,30,20,20,10,20];

rm_artc = [(p(1)-range(1))*fs+1:(p(1)+range(1))*fs,...
           (p(2)-range(2))*fs+1:(p(2)+range(2))*fs,...
           (p(3)-range(3))*fs+1:(p(3)+range(3))*fs,...
           (p(4)-range(4))*fs+1:(p(4)+range(4))*fs,...
           (p(5)-range(5))*fs+1:(p(5)+range(5))*fs,...
           (p(6)-range(6))*fs+1:(p(6)+range(6))*fs,...
           (p(7)-range(7))*fs+1:(p(7)+range(7))*fs,...
           (p(8)-range(8))*fs+1:(p(8)+range(8))*fs];

% remove time periods from the dataset
data_det(rm_artc,:) = [];

%% Estimate the prob. of the switching states given B matrices
% use all the data

% set up
rng(2238)
fs = 250;
y = data_det';
n = size(data_det,2);

% compute var of y on each channel
var_y = zeros(n,1);
for i =1:n
    var_y(i) = cov(data_det(:,i));
end

% parameters setup
k = 4;          % specify the number of oscillators
M = 2;          % specify the number of switching states

% parameters set up
A = zeros(2*k,2*k,M);
Q = zeros(2*k,2*k,M);
R = zeros(n,n,M);

% A matrix
rho = .99;
osc_freq = [0.5,11];

theta1 = (2*pi*osc_freq(1))*(1/fs);
mat1 = [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)];
theta2 = (2*pi*osc_freq(2))*(1/fs);
mat2 = [cos(theta2) -sin(theta2); sin(theta2) cos(theta2)];

% scale of the var
vq = 1;
vr = 10;

% state1
A(:,:,1) = blkdiag(rho*mat2, rho*mat1, rho*mat1, rho*mat1);
Q(:,:,1) = vq*eye(2*k);
R(:,:,1) = vr*diag(var_y);   

% state2
A(:,:,2) = blkdiag(rho*mat2, rho*mat1, rho*mat1, rho*mat1);
Q(:,:,2) = vq*eye(2*k);
R(:,:,2) = vr*diag(var_y);  

% B matrix
B0 = zeros(n,2*k,M);
B0(:,:,1) = B_awake;
B0(:,:,2) = B_unc;

% initial values
inicov = Q(:,:,1)/(1-rho^2);
X_0 = mvnrnd(zeros(size(A,1),1), inicov)';     
V_0 = Q(:,:,1)/(1-rho^2);
pi0 = [1 1]/2;
C = [1-1e-7 1e-7; 1e-7 1-1e-7];

% filter 
[W_j,X_j,V_j,KT] = skf(y',A,B0,Q,R,X_0,C,pi0);
% smoother
[SW,X_RTS,V_RTS,V_cov] = smoother(y',A,B0,Q,R,C,X_j,V_j,W_j,KT);



