% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function [y] = logdet( x )
% y = logdet(x)
% calculates the log determinant of x

[t error] = chol(x);
if error
  error('error');
end
y = sum(log(diag(t))) *2;
