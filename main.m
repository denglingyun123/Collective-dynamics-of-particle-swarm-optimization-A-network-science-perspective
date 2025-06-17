clear all
clc
close all
SearchAgents_no=30; % Number of search agents
Function_name='F1'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)
Max_iteration=500; % Maximum number of iterations  


% Load details of the selected benchmark function������ѡ��ı�׼���Ժ���
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);


[Best_pos1,Best_score1,PSO_curve]=PSO_network(SearchAgents_no,Max_iteration,lb,ub,dim,fobj); %��ʼ�Ż�

figure
semilogy(PSO_curve,'color','[1,0.5,0]','linewidth',2.0,'Marker','o','MarkerIndices',1:50:length(mean(PSO_curve)))    
title('Convergence curve of F_{1}')
xlabel('Iteration');
ylabel('Fitness');
axis tight%�� axis tight�����������������������յ���ʾͼ������ߣ������߽�Ŀհ�
grid off%��ʾ gca ����صĵ�ǰ��������ͼ���������ߡ��������ߴ�ÿ���̶������졣
box on %��ʾ��������Χ������
legend('PSO')


disp('-------------------------------------------------')
display(['������Ӧ��ֵ(Best) : ', num2str(Best_score1)]);


