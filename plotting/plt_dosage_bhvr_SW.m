
% 2022/12/01
% Matlab version: R2021b
% This script is for plotting propofol dosage, behaviroal response, SW

%% plot propofol dosage, behaviroal response, SW

upper = 8500-sum(range)*2;
t2 = (dt:dt:8500);

Fs = 5000;
correct = bhvr.resp_type.correct;
mavg = movmean(correct,10);     %10 is the moving avg window size

figure
% propofol
subplot(4,1,1)
plot(spump.T/Fs,spump.prpfol,'LineWidth',2)
xlim([0 upper])
xline(spump.T(1)/Fs,'LineWidth',1,'LineStyle','--')
xline(spump.T(472)/Fs,'LineWidth',1,'LineStyle','--')
xline(aligntimes(5,4)*60,'LineWidth',2,'Color','k') % LOC
xline(aligntimes(5,6)*60,'LineWidth',2,'Color','k') % ROC
yticks([0,5])
text(aligntimes(5,4)*60+1*60,4.3,"LOC",'Color','k','FontSize',14)
text(aligntimes(5,6)*60+1*60,4.3,"ROC",'Color','k','FontSize',14)
ylabel("Dose (\mug/mL)")
% behavior response
subplot(4,1,2)
plot(bhvr.stim_time.T/Fs,mavg,'LineWidth',2)
xlim([0 upper])
ylabel({'Correct';'response'})
xline(aligntimes(5,4)*60,'LineWidth',2,'Color','k') % LOC
xline(aligntimes(5,6)*60,'LineWidth',2,'Color','k') % ROC
set(gca,'YTick',[0 1],'YLim',[-0.1 1.1]);
% SW - single rhythm model
subplot(4,1,3)
plot(t,SW(:,1,iter+1),'LineWidth',2)
xlim([0 upper])
ylabel('State')
set(gca,'YTick',[0 1],'YLim',[-0.1 1.1]);
xline(aligntimes(5,4)*60,'LineWidth',2,'Color','k')
xline(aligntimes(5,6)*60,'LineWidth',2,'Color','k')
% SW - multiple rhythms model
subplot(4,1,4)
plot(t2,SW_aug(:,1),'LineWidth',2)
xlim([0 upper])
xlabel('Time [s]')
ylabel('State')
set(gca,'YTick',[0 1],'YLim',[-0.1 1.1]);
xline(aligntimes(5,4)*60,'LineWidth',2,'Color','k')
xline(aligntimes(5,6)*60,'LineWidth',2,'Color','k')
