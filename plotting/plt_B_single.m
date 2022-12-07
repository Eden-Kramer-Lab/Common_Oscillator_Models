
% 2022/12/01
% Matlab version: R2021b
% This script is for plotting the B matrices (one oscillator) 
% as a grid / visualizing the network structure. 
% Note: make sure that you have mle_B from single_rhythm_model.m


%% Plot MLE B modes

iter = size(mle_B,4);
input = mle_B(:,:,1,iter);

% electrodes layout
nm = ['','','','',"FP1",'FPz','FP2','','','','',...
    '','','','AF7','AF3','AFz','AF4','AF8','','','',...
    '','F7','F5','F3','F1','Fz','F2','F4','F6','F8','',...
    '','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10',...
    '','T7','C5','C3','C1','Cz','C2','C4','C6','T8','',...
    'TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10',...
    '','P7','P5','P3','P1','Pz','P2','P4','P6','P8','',...
    '','','','PO7','PO3','POz','PO4','PO8','','','',...
    '','','','','O1','Oz','O2','','','',''];

% B1
[mag, phase1] = plt_funcs.B_mag_phase(input);
mx = mag(:,1);
out1 = [0,0,0,0,mx(18),mx(17),mx(48),0,0,0,0;...
        0,0,0,mx(24),mx(12),mx(11),mx(41),mx(54),0,0,0;...
        0,mx(19),mx(13),mx(6),mx(5),mx(2),mx(36),mx(37),mx(42),mx(49),0;...
        0,mx(25),mx(20),mx(7),mx(3),mx(1),mx(33),mx(38),mx(43),mx(55),mx(60);...
        0,mx(14),mx(8),mx(4),mx(9),mx(35),mx(34),mx(39),mx(44),mx(50),0;...
        mx(29),mx(26),mx(21),mx(15),mx(16),mx(10),mx(40),mx(45),mx(51),mx(56),mx(61);...
        0,mx(30),mx(27),mx(22),mx(23),mx(47),mx(53),mx(46),mx(52),mx(57),0;...
        0,0,0,mx(31),mx(28),mx(59),mx(58),mx(62),0,0,0;...
        0,0,0,0,mx(32),mx(64),mx(63),0,0,0,0];

% plot
figure
imagesc(out1)
colorbar
axis off
r=1;
for k=1:9
    for j=1:11
        text(j,k, sprintf('%s',nm(r)),'HorizontalAlignment', 'center')
        r = r+1;
    end
end
% yyaxis left
% ylabel('Left')
% yyaxis right
% ylabel('Right')


% B2
[mag, phase2] = funcs1.B_mag_phase(mle_B(:,:,2,iter));
mx = mag(:,1);
out2 = [0,0,0,0,mx(18),mx(17),mx(48),0,0,0,0;...
        0,0,0,mx(24),mx(12),mx(11),mx(41),mx(54),0,0,0;...
        0,mx(19),mx(13),mx(6),mx(5),mx(2),mx(36),mx(37),mx(42),mx(49),0;...
        0,mx(25),mx(20),mx(7),mx(3),mx(1),mx(33),mx(38),mx(43),mx(55),mx(60);...
        0,mx(14),mx(8),mx(4),mx(9),mx(35),mx(34),mx(39),mx(44),mx(50),0;...
        mx(29),mx(26),mx(21),mx(15),mx(16),mx(10),mx(40),mx(45),mx(51),mx(56),mx(61);...
        0,mx(30),mx(27),mx(22),mx(23),mx(47),mx(53),mx(46),mx(52),mx(57),0;...
        0,0,0,mx(31),mx(28),mx(59),mx(58),mx(62),0,0,0;...
        0,0,0,0,mx(32),mx(64),mx(63),0,0,0,0];

figure
imagesc(out2)
colorbar
axis off
r=1;
for k=1:9
    for j=1:11
        text(j,k, sprintf('%s',nm(r)),'HorizontalAlignment', 'center')
        r = r+1;
    end
end
% yyaxis left
% ylabel('Left')
% yyaxis right
% ylabel('Right')
