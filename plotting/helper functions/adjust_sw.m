
%% note
% this file is for computing SW_aug where it accounts for the remove
% segments in the data

%% removed segments
rm_artc = [(p(1)-range(1))*fs+1:(p(1)+range(1))*fs,...
           (p(2)-range(2))*fs+1:(p(2)+range(2))*fs,...
           (p(3)-range(3))*fs+1:(p(3)+range(3))*fs,...
           (p(4)-range(4))*fs+1:(p(4)+range(4))*fs,...
           (p(5)-range(5))*fs+1:(p(5)+range(5))*fs,...
           (p(6)-range(6))*fs+1:(p(6)+range(6))*fs,...
           (p(7)-range(7))*fs+1:(p(7)+range(7))*fs,...
           (p(8)-range(8))*fs+1:(p(8)+range(8))*fs];

p = [1900,2030,2795,2900,4085,6050,6265,6908];
range = [20,20,40,30,20,20,10,20];

l1 = zeros(length(p),1);
l2 = zeros(length(p),1);
for i=1:length(p)
    l1(i) = (p(i)-range(i))*fs+1;
    l2(i) = (p(i)+range(i))*fs;
end

u1 = zeros(length(p),1);
u2 = zeros(length(p),1);
for i=1:length(p)
    u1(i) = l2(i)+1;
    u2(i) = l1(i);
end
u1 = cat(1,1,u1);
ttotal = size(SW,1) + length(rm_artc);
u2 = cat(1,u2,ttotal);
intv = [0, range*2*fs];

SW_aug = NaN(ttotal,1);
count = 0;
for i=1:length(u1)
    count = count + intv(i);
    for j=u1(i):u2(i)
        SW_aug(j) = SW(j-count,1);
    end
end
