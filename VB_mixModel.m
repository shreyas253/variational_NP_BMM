function [ res ] = VB_mixModel( x,prior,op,seed )
% function [ res ] = VB_mixModel( x,prior,op,seed )
% 
% Performs clustering using Bayesian Mixture models using Variational inference with the specified
% models and priors. The models possible are Von Mises-Fisher mixture
% models (VMM) and Gaussian mixture models (GMM; with Full, Diagonal and Fixed
% covariance). The priors on the GMMs are the corresponding conjugate
% priors and those on the VMM are a VMF distribution and a Gamma distribution (see reference doc). 
% The priors on the weight distribution can be the parametric Dirichlet Distribution (DD) or non parametric
% options such as the Dirichlet process (DP) or the Pitman-Yor process (PYP)
%
% Inputs:   
%   x:          data, N x D
%   op:         object various options
%       op.Pi_Type:      'DP', 'DD' or 'PYP'(include prior.g)
%       op.model_Type:    'GMM-Full', 'GMM-Diag', 'GMM-Fixed' or 'VMM'
%       op.init_Type:    'random' or 'self'(include op.Init_r = initial responsibilities)
%       op.repeats:      no of repeats of the whole EM procedure (e.g. 10 or 20)
%       op.stopCrit:     'freeEnergy' or 'number'(include op.noStop = number of EM iterations)
%       op.max_num_iter: maximum no of EM iterations 
%       op.K:            max number of clusters
%       op.freethresh:   threshold for freeEnergy(e.g. 1e-6)
%       op.reorder:      binary to control whether to reorder the clusters (in descending order of data associated; used in non-parametric methods) after each EM iteration 
%   prior:      object with different prior values depending on model type.
%               The prior parameters used for different op.model_Type are
%       'GMM-Full':     m0, beta0, W0 and v0 (Normal Whisart - see Bishop)
%       'GMM-Diag':     m0, beta0, a0 and b0 (Normal Gamma - see (http://www.albany.edu/~yy298919/realvb.pdf))
%       'GMM-Fixed':    m0, Pres0(params of gaussain modelling mean) and PresMain(fixed/known precision)
%       'VMM':          mu, beta, a and b (see reference doc)
%               The priors for the weight distribution are
%       prior.alpha:    parameter of 'DD', 'DP' or 'PYP'
%       prior.g:        parameter of 'PYP'
%   seed:       seed for random generator
%
% Outputs:
%   res:        clustering results
%       res.z:          cell of found cluster indices for each repeat
%       res.Nk:         cell of cluster allocations(sum of resposibilities) for each repeat
%       res.seed:       seed for random generator (same as input- seed)
%
% By: Shreyas Seshadri (shreyas.sesahdri@aalto.fi), Ulpu Remes and Okko Rasaen, Last update:19.10.2016
% (C) MIT license
% For license terms and references, see README.txt

[N,D] = size(x);
rng(seed);
if strcmp(op.model_Type,'VMM')
    assert(min(sum(power(x,2),2))>1-exp(-20)); % check unit norm
    assert(max(sum(power(x,2),2))<1+exp(-20));
    op.max_kk = round(size(x,2)/2); 
    while isfinite(besseli(size(x,2)/2,op.max_kk+10))
        op.max_kk = op.max_kk+10;
    end
    extra_V.kk = repmat(prior.a./prior.b,1,op.K);
    extra_V.ln_kk = psi(prior.a)-log(prior.b);
    extra_V.kk_approx=extra_V.kk-(prior.a>1)*(1./prior.b);
else
    extra_V = [];
end

for itr = 1:op.repeats
    if strcmp(op.init_Type,'random')
        r = rand(N,op.K);
        r = r./repmat(sum(r,2),1,op.K);
    elseif strcmp(op.init_Type,'self')
        r = op.Init_r;        
    end
    [post,~,extra_V] = postUpdate(x,r,prior,op,extra_V);
    whileCheck =1;
    itr2=0;
    oldFreeEnergy = inf;

    while whileCheck
        itr2 = itr2+1;
        [r,extra_E] = updateR(x,post,op,extra_V); % E step
        if op.reorder
            [extra_E,r] = reorderFE(extra_E,r,op,extra_V);
        end
        [post,extra_M,extra_V] = postUpdate(x,r,prior,op,extra_V); % M step
        extra = structMerge(extra_E,extra_M,extra_V);
        % check stopping
        if strcmp(op.stopCrit,'freeEnergy') 
            newFreeEnergy = freeEnergyCalc(prior,post,r,op,extra); % get free energy
            freeEnergyDiff = (newFreeEnergy - oldFreeEnergy)/newFreeEnergy;
            if ((abs(freeEnergyDiff)<op.freethresh) && (itr2>20)) || (itr2>op.max_num_iter)
                whileCheck = 0;
            end 
            oldFreeEnergy = newFreeEnergy;
        elseif strcmp(op.stopCrit,'number')
            if (itr2 == op.noStop) || (itr2>op.max_num_iter)
                whileCheck = 0;
            end 
        end
    end
    fprintf(['repeat ' num2str(itr) ' took ' num2str(itr2) ' iterations']);
    
    %get results
    [~,res.z{itr}] = max(r,[],2);
    res.Nk{itr} = extra.Nk;
    res.seed = seed;
    res.model.post = post;
    res.model.extra = extra_V;    
end

end

