
% 2022/12/01
% Matlab version: R2021b
% This script is for plotting the B matrices (multiple oscillators) 
% as a grid / visualizing the network structure. 
% Note: make sure that you have mle_B_awake & mle_B_unc from 
% multi_rhythms_model.m

%% Electrodes 
ename = ["FCz", "Fz", "FC1", "C3", "F1", "F3", "FC3", "C5", ...
    "C1", "CPz", "AFz", "AF3", "F5", "T7", "CP3", "CP1",...
    "FPz", "FP1", "F7", "FC5", "CP5", "P3", "P1", "AF7",...
    "FT7", "TP7", "P5", "PO3", "TP9", "P7", "PO7", "O1",...
    "FC2", "C2", "Cz", "F2", "F4", "FC4", "C4", "CP2",...
    "AF4", "F6", "FC6", "C6", "CP4", "P4", "Pz", "FP2",...
    "F8", "T8", "CP6", "P6", "P2", "AF8", "FT8", "TP8",...
    "P8", "PO4", "POz", "FT10", "TP10", "PO8", "O2", "Oz"];

% identify the electrode # to be removed
remove = [4,8,9,10,16,35];

% remove electrode names from the list
ename(remove) = [];

%% Create a new layout

% identify the electrodes to be removed
remove_name = ["C3","C5","C1","CPz","CP1","Cz"];

% original electrodes layout
nm = ['','','','',"FP1",'FPz','FP2','','','','',...
    '','','','AF7','AF3','AFz','AF4','AF8','','','',...
    '','F7','F5','F3','F1','Fz','F2','F4','F6','F8','',...
    '','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10',...
    '','T7','C5','C3','C1','Cz','C2','C4','C6','T8','',...
    'TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10',...
    '','P7','P5','P3','P1','Pz','P2','P4','P6','P8','',...
    '','','','PO7','PO3','POz','PO4','PO8','','','',...
    '','','','','O1','Oz','O2','','','',''];

% replaced the removed electrodes by ""
for i=1:length(nm)
    for j=1:length(remove_name)
        if nm(i) == remove_name(j)
            nm(i) = "";
        end
    end
end

%% Plot MLE B modes -- awake state

iter = size(mle_B_awake,3);
input = mle_B_awake(:,:,iter);
fq = [.5,11];

nodes = size(input,1);
osci = size(input,2)/2;

% compute magnitudes & phases of each oscillator in B matrices
[mag,phase] = plt_funcs.B_mag_phase(input);

% magnitudes layout
out_mag = zeros(9,11,osci);
for i=1:osci
    out_mag(:,:,i) = plt_funcs.layout(ename,mag(:,i));
end

% plot magnitude
upd = max(max(max(mag)));
total = osci;
figure
tiledlayout(1,osci,'Padding','compact','TileSpacing','compact');
for i=1:osci
    nexttile
    imagesc(out_mag(:,:,i))
    colorbar
    caxis([0 upd])
    axis off
    r=1;
    for k=1:9
        for p=1:11
            text(p,k, sprintf('%s',nm(r)),'HorizontalAlignment', 'center')
            r = r+1;
        end
    end
    title(sprintf('Oscillator%g',i))
end
sgtitle(sprintf('B matrix at %g & %g Hz - Magnitude',fq(2),fq(1)))

%% Plot MLE B modes -- unconscious state

iter = size(mle_B_unc,3);
input = mle_B_unc(:,:,iter);
fq = [.5,11];

nodes = size(input,1);
osci = size(input,2)/2;

% compute magnitudes & phases of each oscillator in B matrices
[mag,phase] = plt_funcs.B_mag_phase(input);

% magnitudes layout
out_mag = zeros(9,11,osci);
for i=1:osci
    out_mag(:,:,i) = plt_funcs.layout(ename,mag(:,i));
end

% plot magnitude
upd = max(max(max(mag)));
total = osci;
figure
tiledlayout(1,osci,'Padding','compact','TileSpacing','compact');
for i=1:osci
    nexttile
    imagesc(out_mag(:,:,i))
    colorbar
    caxis([0 upd])
    axis off
    r=1;
    for k=1:9
        for p=1:11
            text(p,k, sprintf('%s',nm(r)),'HorizontalAlignment', 'center')
            r = r+1;
        end
    end
    title(sprintf('Oscillator%g',i))
end
sgtitle(sprintf('B matrix at %g & %g Hz - Magnitude',fq(2),fq(1)))

