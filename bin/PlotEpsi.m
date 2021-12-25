function PlotEpsi(Epsi)

figure('Color','w', 'Position',[479         703        1095         346]);

subplot(1,2,1);
title(['Betas'],'FontWeight','bold','FontSize',14);
bar([0 Epsi.betas], 'r');
xlabel('Ws','FontWeight','bold','FontSize',14);
ylabel(['Betas'],'FontWeight','bold','FontSize',14);

subplot(1,2,2);
title(['Lambdas'],'FontWeight','bold','FontSize',14);
bar(Epsi.lambdas, 'g');
xlabel('Ws','FontWeight','bold','FontSize',14);
ylabel(['Lambdas'],'FontWeight','bold','FontSize',14);

