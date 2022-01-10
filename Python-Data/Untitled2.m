
classLDA=classify(data(:,1:end-1),data(:,1:end-1),data(:,end),'linear');
CA=classperf(testlabels,classLDA);