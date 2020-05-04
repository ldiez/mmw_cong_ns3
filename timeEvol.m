close all;
clear;
clc

bw = 20;
dist = 30;
tail = sprintf('_%dMbps_%04d', bw, dist);
path = 'traces';
d = dlmread(sprintf('%s/static%s.txt', path, tail), ',');
d2 = dlmread(sprintf('%s/DlRxPhy%s.txt',path,tail), ',');
initTime = .5;
totalTime = 30;

d = d(find(d(:, 2) < totalTime & d(:, 2) > initTime), :);
d2 = d2(find(d2(:, 1) < totalTime & d2(:, 1) > initTime), :);

step = 1;
Mm = [d(1:step:end, 2) d(1:step:end, 3)/1024 d(1:step:end, 4) d(1:step:end, 5)];
Mp = [d2(1:step:end, 1) d2(1:step:end, 4) d2(1:step:end, 2)/1024];
figure;
subplot(5,1,1)
plot(Mp(:,1), Mp(:,2))
mean(Mp(:,2))
title('SINR (dB)')

subplot(5,1,2)
plot(Mp(:,1), Mp(:,3))
title('Phy bytes sent')

subplot(5,1,3)
plot(Mm(:,1), Mm(:,2));
title('RLC buffer occupancy')

subplot(5,1,4)
plot(Mm(:,1), Mm(:,3));
title('Delay')

subplot(5,1,5)
plot(Mm(:,1), Mm(:,4));
title('IaT')

% dlmwrite(sprintf('congestion-control/figs/data/phy_d%d_b%d.txt', dist, bw),Mp, 'delimiter', ' ');
% dlmwrite(sprintf('congestion-control/figs/data/mac_d%d_b%d.txt', dist, bw),Mm, 'delimiter', ' ');

% dlmwrite(sprintf('congestion-control/figs/data/phy_d%d_b%d_mw.txt', dist, bw),Mp, 'delimiter', ' ');
% dlmwrite(sprintf('congestion-control/figs/data/mac_d%d_b%d_mw.txt', dist, bw),Mm, 'delimiter', ' ');