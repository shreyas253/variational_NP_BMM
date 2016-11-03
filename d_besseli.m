function bes = d_besseli(nu,kk)
% function bes = d_besseli(nu,kk) evaluates d/d_x ln(I_nu(x)) at points kk
%
% Inputs:   
%   nu:         besseli order (D/2-1)
%   kk:         points where function is evaluated 1 x T (E_q(kk))
%
%
% Outputs:
%   bes:        d/d_x ln(I_nu(x)) evaluated at points kk or upper bound      
%
% By: Ulpu Remes, Last update: 19.10.2016  
    
    try
        bes=besseli(nu+1,kk)./(besseli(nu,kk)+exp(-700))+nu./kk;
        assert(min(isfinite(bes)))
    catch me
        bes=sqrt(1+power(nu,2)./power(kk,2)); % approximate with upper bound
    end
end