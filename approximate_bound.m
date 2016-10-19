function bound=approximate_bound(data,mu,kk,ln_kk,kk_approx)
% function bound=approximate_bound(data,mu,kk,ln_kk,kk_approx)
% 
% calculate lower bound on the expected state likelihoods
% Inputs:   
%   data:       N x D
%   mu:         D x T
%   kk:         1 x T, E_q(kk)
%   ln_kk:      1 x T, E_q(ln_kk)
%   kk_approx:  1 x T  
%
% Outputs:
%   bound:      
% By: Ulpu Remes, Last update: 19.10.2016
    
    nu=size(mu,1)/2-1;
    
    % calculate upper bound on the bessel term:
    
    bes_bound=log(besseli(nu,kk_approx)+exp(-700))+d_besseli(nu,kk_approx).*(kk-kk_approx);
    
    % calculate likelihood bound:
    
    c_k = nu*ln_kk-(nu+1)*log(2*pi)-bes_bound; % normalisation term
    
    distances = data*mu; % N x T
    bound = bsxfun(@plus,bsxfun(@times,distances,kk),c_k);   
end