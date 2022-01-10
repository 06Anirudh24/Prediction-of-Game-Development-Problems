
% for i=1:4
%     in1{i}=find(x(:,end)==i-1);
% end

% g1=ones(1,size(in1{1},1));
% g2=2*ones(1,size(in1{2},1));
% g3=3*ones(1,size(in1{3},1));
% g4=4*ones(1,size(in1{4},1));
% 
% for j=1:7
% fname=strcat(num2str(j),'.csv');
% data=csvread(fname);
% p1=0;
% for i=1:size(data,2)-1
% x=[data(in1{1},i);data(in1{2},i);data(in1{3},i);data(in1{4},i)]' ;
% g=[g1,g2,g3,g4]; 
% p1(i)=anova1(x,g,'off');   
% end
% inn=find(p1<=0.05);
% datan=data(:,inn);
% datan(:,end+1)=data(:,end);
% fname=strcat(num2str(7+j),'.csv');
% csvwrite(fname,datan);
% end

% for i=1:7
%  fname=strcat(num2str(i),'.csv');
%  data=csvread(fname); 
%  [coeff,score,latent,tsquared] = pca(data(:,1:end-1));
%  latent(:,2)=latent(:,1)/sum(latent(:,1));
%  latent(:,2)=latent(:,2)*100;
%  in=find(latent(:,2)>0.1);
%  datan=score(:,in);
%  datan(:,end+1)=data(:,end);
%  fname=strcat(num2str(14+i),'.csv');
%  csvwrite(fname,datan);
%  sum(latent(in,2))
% end
% 
