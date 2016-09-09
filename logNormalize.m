% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function y = logNormalize( x )
%LOGNORMALIZE 
% x is a 2 D matrix to be normalized along dim 2
% y(:,i) = exp(x(:,i)) / sum(exp(x(:,i))) 

[d,k] = size(x);
x_max = max(x, [], 2);
x_max(x_max==-inf) = 0;

lnDr = x_max + log(sum(exp(x-repmat(x_max,1,k)), 2));

y = exp(x-repmat(lnDr,1,k));

end

