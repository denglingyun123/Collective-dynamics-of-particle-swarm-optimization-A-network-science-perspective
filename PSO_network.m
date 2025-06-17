%% Network modeling of PSO
%  Developed in MATLAB R2018a   
%  Author : Lingyun Deng, Sanyang Liu                                                         
%                                                                                                     
%         e-Mail: lingyundeng@stu.xidian.edu.cn                                                            
%                 syliu@xidian.edu.cn 
                                                                                                                                                                                                        
%  Main paper:                                                                                        
%  Lingyun Deng,Sanyang Liu. Collective dynamics of particle swarm optimization: A network science perspective
%  Journal name= Physica A: Statistical Mechanics and its Applications, 


function [Best_pos,Best_score,curve]=PSO_network(NP,Max_iter,lb,ub,dim,fobj)

%% Parameter setup
wMax = 0.9;
wMin = 0.4;
c1 = 2;       
c2 = 2;      
% Vmax=30;
% Vmin=-30;
if(max(size(ub)) == 1)
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
end

Vmax = (ub - lb) .* 0.2;
Vmin  = -Vmax;


%% Initialization
Range = ones(NP,1)*(ub-lb);
pop = rand(NP,dim).*Range + ones(NP,1)*lb;    % Initialization of the population
V = rand(NP,dim).*(Vmax-Vmin) + Vmin;                 % Initialization of the velocity
fitness = zeros(NP,1);
for i=1:NP
    fitness(i,:) = fobj(pop(i,:));                         % Fitness evaluation
end

%% Obtain the personal best and global best individuals
[bestf, bestindex]=min(fitness);
gbest=pop(bestindex,:);   % Global best individual
pbest=pop;                % Personal best individual
fitnesspbest=fitness;              % Personal best fitness value
fitnessgbest=bestf;               % Global best fitness value

%% Record the index numbers corresponding to the personal best and global best individuals
for i=1:NP
    pbest_index(i)=i;
end
gbest_index=bestindex;

%% Record the index numbers corresponding to the source node and target node.
source_index=[];
target_index=[];


iter = 0;
while( (iter < Max_iter ))
    
    w = wMax - iter .* ((wMax - wMin) / Max_iter);
    iter = iter+1;                     
    
    %% Create an edge table, 
    % populating the source_index and target_index fields with the individuals involved in the position updates.
    for i=1:NP
        if iter==1
            source_index=[source_index;NP*iter+i];
            target_index=[target_index;pbest_index(i)];
        
            source_index=[source_index;NP*iter+i];
            target_index=[target_index;gbest_index];
            
            source_index=[source_index;NP*iter+i];
            target_index=[target_index;i];
            
        else
            source_index=[source_index;NP*iter+i];
            target_index=[target_index;pbest_index(i)];
        
            source_index=[source_index;NP*iter+i];
            target_index=[target_index;gbest_index];
            
            source_index=[source_index;NP*iter+i];
            target_index=[target_index;NP*(iter-1)+i];
            
            source_index=[source_index;NP*iter+i];
            target_index=[target_index;NP*(iter-2)+i];
        end
    end
    
    Edge_table=[source_index,target_index];
    
    %% Remove the ¡°self-loops¡± from the edge table.
    Edge_table=unique(Edge_table, 'rows');
    
   
    for i=1:NP
        
        % Velocity update
        V(i,:) = w*V(i,:) + c1*rand*(pbest(i,:) - pop(i,:)) + c2*rand*(gbest - pop(i,:));
        
        % Position update
        pop(i,:)=pop(i,:)+V(i,:);
        for j=1:dim
            if pop(i,j)>ub(j)
                pop(i,j)=ub(j);
            end
            if pop(i,j)<lb(j)
                pop(i,j)=lb(j);
            end
        end
        
        % Fitness value
        fitness(i,:) =fobj(pop(i,:));
        
        % Update the personal best individual
        if fitness(i) < fitnesspbest(i)
            pbest(i,:) = pop(i,:);
            fitnesspbest(i) = fitness(i);
            pbest_index(i)=iter*NP+i;  % Update the index of the personal best individual
        end
        
        % Update the global best individual
        if fitness(i) < fitnessgbest
            gbest = pop(i,:);
            fitnessgbest = fitness(i);
            gbest_index=iter*NP+i;  % Update the index of the global best individual
        end
    end
    curve(iter) = fitnessgbest;
end

Best_pos = gbest;
Best_score = fitnessgbest;


%% Claim that the type of the network is undircted
[rows,~]=size(Edge_table);
for i=1:rows
    type_edge{i}='undirected';
end

%% Save the edge table
table_csv = table(Edge_table(:,1),Edge_table(:,2),type_edge');
table_csv.Properties.VariableNames = {'source','target','type'};
writetable(table_csv,"C:\Users\dengl\Desktop\Edge.csv")  %******Change it according to your need********

end



