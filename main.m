clear all
clc
close all
SearchAgents_no=30; % Number of search agents
Function_name='F1'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)
Max_iteration=500; % Maximum number of iterations  


% Load details of the selected benchmark function—加载选择的标准测试函数
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);


[Best_pos1,Best_score1,PSO_curve]=PSO_network(SearchAgents_no,Max_iteration,lb,ub,dim,fobj); %开始优化

figure
semilogy(PSO_curve,'color','[1,0.5,0]','linewidth',2.0,'Marker','o','MarkerIndices',1:50:length(mean(PSO_curve)))    
title('Convergence curve of F_{1}')
xlabel('Iteration');
ylabel('Fitness');
axis tight%用 axis tight命令可以让坐标轴调整到紧凑地显示图像或曲线，不留边界的空白
grid off%显示 gca 命令返回的当前坐标区或图的主网格线。主网格线从每个刻度线延伸。
box on %显示坐标区周围的轮廓
legend('PSO')


disp('-------------------------------------------------')
display(['最优适应度值(Best) : ', num2str(Best_score1)]);


