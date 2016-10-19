function bes = d_besseli(nu,kk)
% function bes = d_besseli(nu,kk)
% 
% Description?
% kk here can exceed max_kk so need to approximate sometimes
%
% Inputs:   
%   nu:         N x D
%   kk:         1 x T, E_q(kk)
%
% Outputs:
%   bes:              
%
% By: Ulpu Remes, Last update: 19.10.2016  
    
    try
        bes=besseli(nu+1,kk)./(besseli(nu,kk)+exp(-700))+nu./kk;
        assert(min(isfinite(bes)))
    catch me
        bes=sqrt(1+power(nu,2)./power(kk,2));
    end
end