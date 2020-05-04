clear;
clc

bws = [10, 20, 30, 40, 50, 100, 200, 500, 1000];
dists = [10, 20, 30, 40, 50];
path = 'traces';

corrDel = [];
corrIat = [];

for bw = bws
    auxCorrDel = [];
    auxCorrIat = [];
    for dist = dists
        tail = sprintf('_%dMbps_%04d', bw, dist);
        fn = sprintf('%s/static%s.txt', path, tail)
        d = dlmread(fn, ',');
        data = d(: , [3,4,5]);
        
        aux = corr(data);
        aux = aux(1,2:3);
        auxCorrDel = [auxCorrDel aux(1)];
        auxCorrIat = [auxCorrIat aux(2)];        
    end
    corrDel = [corrDel; auxCorrDel];
    corrIat = [corrIat; auxCorrIat];
end    

len = length(corrDel(:, 1));
xlabels = {'10Mbps','20Mbps','30Mbps','40Mbps','50Mbps','100Mbps','200Mbps','500Mbps','1000Mbps'};
figure;
bar ([1:len], abs(corrDel));
set(gca,'xticklabel',xlabels)
legend('10m', '20m', '30m', '40m', '50m')
title ('Buffer-delay correlation - default schelduler')
saveas(gcf,'movStd10DelCorrMaxWeight.png')


figure;
bar ([1:len], abs(corrIat));
set(gca,'xticklabel',xlabels)
legend('10m', '20m', '30m', '40m', '50m')
title ('Buffer-IAT correlation - default schelduler')
saveas(gcf,'movStd10IatMaxWeight.png')

% dlmwrite('congestion-control/figs/data/del_corr.txt',abs(corrDel), 'delimiter', ' ');
% dlmwrite('congestion-control/figs/data/iat_corr.txt',abs(corrIat), 'delimiter', ' ');