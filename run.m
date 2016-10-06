% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt

%close all; clear;

%% load your data

% data = (); % as N*D where N=no of data, and D=dimensions

[N,D]=size(data);


%% GMM 
% currently DDGMM, DPGMM and PYPGMM
prior.alpha = 1;
op.Pi_Type = 'PYP'; % DP, DD or PYP(include prior.g)
prior.g=0.5;
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


data=bsxfun(@rdivide,data,sqrt(sum(power(data,2),2))); % normalise to unit length

vmm_op=op;
vmm_op.freetresh=0.001; % update threshold value because definitions are different

% mean parameter prior:
vmm_prior.mu=sum(data)/norm(sum(data)); % assume data points close to mean
vmm_prior.beta=0.05; % do not trust mean parameter too much

% concentration parameter prior:

vmm_prior.a=1; % assume unconcentrated data
vmm_prior.b=0.01; % but do not trust that too much

switch op.Pi_Type

    case 'DD'
    
    for rr=1:op.repeats
        vmm_prior.alpha=prior.alpha; 
        resultVMM{rr}=train_variational_dirichlet(data,vmm_prior,vmm_op);
    end
        
    case 'DP'
        
    for rr=1:op.repeats
        vmm_prior.s_1=prior.alpha; % alpha of earlier
        vmm_prior.s_2=0; % zero when want to use DP(g of earlier)
        resultVMM{rr}=train_variational_pitman_yor(data,vmm_prior,vmm_op);
    end
        
    case 'PYP'
        
    for rr=1:op.repeats
        vmm_prior.s_1=prior.alpha;
        vmm_prior.s_2=prior.g;
        resultVMM{rr}=train_variational_pitman_yor(data,vmm_prior,vmm_op);
    end
    
    otherwise
    error(['unknown weight distribution prior: ' op.init_Type])
end
foundClustersV=cell(1,op.repeats);
for rr=1:op.repeats
[~,foundClustersV{rr}]=max(resultVMM{rr}.z,[],2); %are the found cluters
end

