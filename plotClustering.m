function plotClustering( data,z,titleStr )

figure;
colors = {'k','b','r','g','y','cy','m'};
zVals = unique(z)';
plotCount=0;
for k = zVals
    plotCount = plotCount+1;
    inds = (z == k);
     if plotCount>length(colors)
        colors = [colors rand(3,1)];
    end
    plot(data(inds,1),data(inds,2),'.','color',colors{plotCount});hold on;
end
hold off;
title(titleStr);

end

