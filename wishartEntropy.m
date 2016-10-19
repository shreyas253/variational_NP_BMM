% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function [ entropyWish ] = wishartEntropy( W,v )
%WISHARTENTROPY 
% inputs - K wishart distribution paramters W(:,:,k) and v(k), for k=1:K
% output - K array with etropy of each wishart distibution
% ignores constabt term
[~,D,K] = size(W);

for k = 1:K
    entropyWish(k) = (D+1)*sum(log(diag(chol(W(:,:,k))))) ... log(det(A)) = 2 * sum(log(diag(chol(A))))
        + D*(D-1)*0.25*log(pi)*sum(gammaln((v(k)+1-(1:D))/2)) ...
        + 0.5*(v(k)+D-1)*sum(psi((v(k)+1-(1:D))/2));
end

end

