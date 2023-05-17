
% 2022/12/01
% Matlab version: R2021b
% This script is for plotting the B matrices (one oscillator) 
% as a grid / visualizing the network structure. 
% Note: make sure that you have mle_B from single_rhythm_model.m
close all

%% Plot the data

t = (1:T)/fs;              %time axis

figure
plot(t,y)


%% Empirical and Theoretical spectra
% relies on Hugo's get_theoretical_psd

params = struct('Fs',fs,'tapers',[3000, 20]);
[S,f] = mtspectrumc(y',params);

[H_tot, H_i]=get_theoretical_psd(f,fs,osc_freq,rho,vq);
Stheo = H_tot+vr/fs;

figure
plot(f,10*log10(S),'k')
hold on
plot(f,10*log10(Stheo),'r','linewidth',2)
hold off


%% Plot MLE B modes

iter = size(mle_B,4);
input = mle_B(:,:,1,iter);

toplot = mle_B(:,:,:,iter);
% toplot = B0;

figure
for j=1:M
    [mag, phase1] = plt_funcs.B_mag_phase(toplot(:,:,j));
    for i=1:k
        mx = mag(:,i);
        out1 = [mx(1),mx(3);...
                mx(2),mx(4)];
        subplot(M,k,sub2ind([k,M],i,j))
        imagesc(out1)
        colorbar
%         axis off
        
        r=1;
        for l=1:2
            for m=1:2
                text(l,m, sprintf('%s',r),'HorizontalAlignment', 'center')
                r = r+1;
            end
        end

    end
end

%%

figure
plot(t,SW(:,:,iter+1),'LineWidth',2)
ylabel('State')
set(gca,'YTick',[0 1],'YLim',[-0.1 1.1]);