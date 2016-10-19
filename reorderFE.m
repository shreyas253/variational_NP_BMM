% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function [  extra,r  ] = reorderFE( extra,r,op,extra_V )
%REORDERFE 
% reorder all variables according to the descending Nk
[~,newI] = sort(extra.Nk,'descend');

r = r(:,newI);

extra.E_lnPik = extra.E_lnPik(newI);
if  strcmp(op.model_Type,'GMM-Full') || strcmp(op.model_Type,'GMM-Diag')
    if length(extra.E_lnPresk)>1
        extra.E_lnPresk = extra.E_lnPresk(newI);
    end
end
if strcmp(op.model_Type,'VMM')
    extra_V.kk = extra_V.kk(newI);
    extra_V.ln_kk = extra_V.ln_kk(newI);
    extra_V.kk_approx = extra_V.kk_approx(newI); 
end


end

