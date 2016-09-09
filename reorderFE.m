% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function [  extra,r  ] = reorderFE( extra,r,op )
%REORDERFE 
% reorder all variables according to the descending Nk
[~,newI] = sort(extra.Nk,'descend');

r = r(:,newI);

extra.E_lnPik = extra.E_lnPik(newI);
if  ~(strcmp(op.cov_Type,'Fixed'))
    if length(extra.E_lnPresk)>1
        extra.E_lnPresk = extra.E_lnPresk(newI);
    end
end


end

