# Network modeling of PSO
%  Main paper:                                                                                        
%  1.Lingyun Deng,Sanyang Liu. Collective dynamics of particle swarm optimization: A network science perspective
%  Journal name= Physica A: Statistical Mechanics and its Applications

%  2.Lingyun Deng,Sanyang Liu. Unlocking new potentials in evolutionary computation with complex network insights: A brief survey
%  Journal name= Archives of Computational Methods in Engineering


REMARK:
1. When you applying PSO to solve a problem, you should run the main.m and you will obtain an eage table that appears in the predefined path. Then, you should use the specialized software like Gephi to analyze the edge table. This software can calculate many important metrics of the network such as average path length, clustering coefficient, etc. More importantly, users can export the degree sequence from the software.
2. We use the Matlab to obtain a edge table. Then the degree sequence is exported from the Gephi (https://gephi.org/). Finally, we use the Python to fit the degree sequence since it has some powerful packages such as Scipy and powerlaw.
