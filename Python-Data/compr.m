% boxplot(x);
% set(gca, 'XTick',1:5, 'XTickLabel',tec(1:5));
% grid on
% set(gca, 'XTickLabelRotation', 60);
% set(gca,'FontSize',10,'FontWeight','bold');
% ylabel('AUC');

% for i=1:8
%   y(1:7,5*(i-1)+1:5*i)=x(7*(i-1)+1:7*i,:); 
% end
% y=y';

% boxplot(y);
% set(gca, 'XTick',1:7, 'XTickLabel',tec(6:12));
% grid on
% set(gca, 'XTickLabelRotation', 60);
% set(gca,'FontSize',10,'FontWeight','bold');
% ylabel('AUC');

% for i=1:2
%    for j=1:28
%       y(5*(j-1)+1:5*j,i)=x(28*(i-1)+j,:); 
%    end
%     
% end

boxplot(y);
set(gca, 'XTick',1:2, 'XTickLabel',tec(13:14));
grid on
set(gca, 'XTickLabelRotation', 60);
set(gca,'FontSize',10,'FontWeight','bold');
ylabel('AUC');