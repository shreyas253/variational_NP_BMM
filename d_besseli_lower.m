function bes = d_besseli_lower(nu,kk)
% function bes = d_besseli_lower(nu,kk) evaluates d/d_x ln(I_nu(x)) at
% points kk
%
% Inputs:   
%   nu:         besseli order (D/2-1)
%   kk:         points where function is evaluated 1 x T (E_q(kk))
%
%
% Outputs:
%   bes:        d/d_x ln(I_nu(x)) evaluated at points kk or lower bound      
%
% By: Ulpu Remes, Last update: 3.11.2016  
    
    try
        bes=besseli(nu+1,kk)./(besseli(nu,kk)+exp(-700))+nu./kk;
        assert(min(isfinite(bes)))
    catch me
        bes=kk./(nu+1+sqrt(kk.^2+(nu+1)^2))+nu./kk; % approximate with lower bound
    end
end