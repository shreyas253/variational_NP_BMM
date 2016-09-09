% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function [ r,extra ] = updateR( x,post,op )
%UPDATER Summary of this function goes here
%   Detailed explanation goes here

[N,D] = size(x);
K = size(post.m,2);

rho = zeros(N,K);
rho2= rho;


for k=1:K
    if strcmp(op.Pi_Type,'DP') ||  strcmp(op.Pi_Type,'PYP')
        if k < K
          E_lnPik = ...
              psi(post.gamma(1,k)) - psi(sum(post.gamma(:,k),1)) ...
              + sum(psi(post.gamma(2,[1:k-1])) - psi(sum(post.gamma(:,[1:k-1]),1)), 2);
        else
          E_lnPik = sum(psi(post.gamma(2,[1:k-1])) ...
                                             - psi(sum(post.gamma(:,[1:k-1]),1)), 2);
        end
    elseif strcmp(op.Pi_Type,'DD')
        E_lnPik = psi(post.DDalpha(k)) - psi(sum(post.DDalpha));
    end
    extra.E_lnPik(k) = E_lnPik;
    if strcmp(op.cov_Type,'Full')
      psiSum = sum(psi( (post.v(k)+1 - (1:D))*0.5 )); % 1*1
      E_lnPresk = psiSum + D*log(2) + logdet(post.W(:,:,k));%1*1
      
      d = x - repmat(post.m(:,k),1,N)'; %N*D
      E_mukPresk_term = D/post.beta(k) + post.v(k)*sum(d.*(post.W(:,:,k)*d')',2);
      
      E_lnx = E_lnPresk/2 -0.5*D*log(2*pi) - E_mukPresk_term/2; % 1*N
      extra.E_lnPresk(k) = E_lnPresk;  
    elseif strcmp(op.cov_Type,'Diag')
%       checkmat = hp_posterior.eta(c)+log(hp_posterior.B(:,c));
%       check = checkmat(1)<=0 || checkmat(2)<=0; 
      E_lnPresk = sum(psi(post.a(k))+log(post.b(:,k))); %1*1
      d = x - repmat(post.m(:,k),1,N)'; %N*D
      prod = repmat(post.a(k) * post.b(:,k),1,N)';%N*D
      E_mukPresk_term = sum((prod .* d.^2)+(1/post.beta(k)),2); %N*1       
      E_lnx = 0.5*E_lnPresk - 0.5*E_mukPresk_term;
      extra.E_lnPresk(k) = E_lnPresk;
    elseif strcmp(op.cov_Type,'Fixed')
      d = x - repmat(post.m(:,k),1,N)'; %N*D
      E_muk_term = sum(d.*(op.PresMain*d')',2)+D;  
      E_lnx = -0.5*E_muk_term;
    end
    rho(:,k) = E_lnx + E_lnPik;%N*1
    %rho2(:,k) = E_lnPik - 0.5*E_mukPresk_term;
end
r = logNormalize(rho);
extra.Nk = sum(r,1);

%[~,zq] = max(r,[],2);
%length(unique(zq))
