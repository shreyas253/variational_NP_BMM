% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt

close all; clear;

%% load your data
data = rand(1000,10);
[N,D] = size(data);

%% GMM 
% currently DDGMM, DPGMM and PYPGMM
prior.alpha = 1;
op.Pi_Type = 'DD'; % DP, DD or PYP(include prior.g)
op.cov_Type = 'Fixed'; % can be 'Full' and 'diag' (check appropriate priors required)
prior.m0 = mean(data,1);
prior.Pres0 = (1/0.05)*eye(D);%inv(cov(data)); %prior cov of the gaussian used to model the means
op.PresMain = (1/1e-3)*eye(D);%inv(cov(data)); % known covariance of the GMMs
op.init_Type = 'random'; % initilization
op.repeats = 5;% how many repaeats
op.K=3;% truncation limit
op.stopCrit = 'freeEnergy'; %'number' of runs or 'freeEnergy'
op.freethresh = 1e-6;
op.max_num_iter = 400;
op.reorder = 1; % roderder (usualy used for NP methods)
resultGMM = VB_gmms(data,prior,op);
foundClustersG = resultGMM.z; %are the found cluters

%% VMM
% currently DPVMFMM and PYPVMFMM 

likelihood_thd = 0.001;
init='random';
init_solution=[];
a_1=1; % concentration parameter
% this is how to determine:
% ceil_kk=round(size(data1,2)/2); 
% while isfinite(besseli(size(data1,2)/2,ceil_kk+10))
%     ceil_kk=ceil_kk+10;
% end
ceil_kk=705; % this when 130-dimensional data
data1 = data./repmat(sqrt(sum(abs(data).^2,2)),1,size(data,2));
% mean parameter prior:
mean_o=sum(data1)/norm(sum(data1)); % assume data points close to mean
beta_o=0.05; % do not trust mean parameter too much

% concentration parameter prior:
a_o=1; % assume unconcentrated data
b_o=0.01; % but do not trust that too much

T=3; %truncation limit
m_o=repmat(mean_o,T,1)';

num_iterations=500;
num_runs=5;%no of repeats
foundClustersV = cell(num_runs,1);
for rr=1:num_runs
    s_1=1; % alpha of earlier
    s_2=0; % zero when want to use DP(g of earlier)
    resultVMM{rr}=train_variational_pitman_yor(data1,T,ceil_kk,s_1,s_2,m_o,beta_o,a_o,b_o,num_iterations,likelihood_thd,init,1);
    [~,foundClustersV{rr}]=max(resultVMM{rr}.q_z,[],2); %are the found cluters
 end

