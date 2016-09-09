% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function [ freeEnergy,term1 , term2 , term3 , term4,term4_1,term4_2 ] = freeEnergyCalc( prior,post,r,op,extra )

[N,K] = size(r);
D = length(prior.m0);

%% term1 = E_q[ln(p(X|Z,mu,pres))]
if strcmp(op.cov_Type,'Full')
    term1_allk = zeros(K,1);
    xkBarMinusmk = extra.xkBar - post.m;
    for k = 1:K
        term1_allk(k) = 0.5*extra.Nk(k)*extra.E_lnPresk(k) ...
            - extra.Nk(k)*D/post.beta(k) ...
            - post.v(k)*trace(extra.SkProdNk{k}*post.W(:,:,k)) ...
            - extra.Nk(k)*post.v(k)*xkBarMinusmk(:,k)'*post.W(:,:,k)*xkBarMinusmk(:,k);                        
    end
    term1 = sum(term1_allk);
elseif strcmp(op.cov_Type,'Diag')
    term1_allk = zeros(K,1);
    xkBarMinusmk = extra.xkBar - post.m;
    for k = 1:K
        term1_allk(k) = 0.5*extra.Nk(k)*extra.E_lnPresk(k) ...
            - extra.Nk(k)*D/post.beta(k) ...
            - post.a(k)*sum(extra.SkProdNk{k}.*post.b(:,k)) ...
            - extra.Nk(k)*post.a(k)*sum(post.b(:,k).*xkBarMinusmk(:,k).^2);                        
    end
    term1 = sum(term1_allk);
 elseif strcmp(op.cov_Type,'Fixed')
    term1_allk = zeros(K,1);
    xkBarMinusmk = extra.xkBar - post.m;
    for k = 1:K
        term1_allk(k) = ...
            - trace(extra.SkProdNk{k}*op.PresMain) ...
            - extra.Nk(k)*xkBarMinusmk(:,k)'*op.PresMain*xkBarMinusmk(:,k);                        
    end
    term1 = sum(term1_allk);
end

%% term2 = E_q[ln(p(Z|X))] - E_q[ln(q(Z|X))]
term2_1 = sum(sum( r.*repmat(extra.E_lnPik,N,1) ));
%tmp = r.*log(r);
tmp = r.*log(r+exp(-700));
% [tmpLoc_r,tmpLoc_c] = find(isnan(tmp));
% tmp(tmpLoc_r,tmpLoc_c) = 0;
term2_2 = sum(sum( tmp));
term2 = term2_1 - term2_2;

%% term3 = E_q[ln(p(pi))] - E_q[ln(q(pi))]
if strcmp(op.Pi_Type,'DP')
    %does not vary -> 'gammaln(1+prior.alpha) - gammaln(prior.alpha)'
    term3_all = -gammaln(sum(post.gamma,1))+gammaln(post.gamma(2,:))+gammaln(post.gamma(1,:)) ...
        - (psi(post.gamma(1,:))-1).*(psi(post.gamma(1,:)) - psi(sum(post.gamma,1))) ...
        + (prior.alpha-psi(post.gamma(2,:))).*(psi(post.gamma(2,:)) - psi(sum(post.gamma,1)));
    term3 = sum(term3_all);
elseif strcmp(op.Pi_Type,'PYP')
    term3_all = -gammaln(sum(post.gamma,1))+gammaln(post.gamma(2,:))+gammaln(post.gamma(1,:)) ...
        + (1-prior.g-psi(post.gamma(1,:))).*(psi(post.gamma(1,:)) - psi(sum(post.gamma,1))) ...
        + (prior.alpha-psi(post.gamma(2,:))+((1:K-1)*prior.g)).*(psi(post.gamma(2,:)) - psi(sum(post.gamma,1)));
    term3 = sum(term3_all);    
elseif strcmp(op.Pi_Type,'DD')
    %    log_cAlpha0 = gammaln(K*hp_prior.alpha) - (K*gammaln(hp_prior.alpha));
    log_cAlpha = gammaln(sum(post.DDalpha))- sum(gammaln(post.DDalpha));
    term3 = (prior.alpha-1)*sum(extra.E_lnPik) - log_cAlpha - sum((post.DDalpha -1).*(extra.E_lnPik));
end

%% term4 = E_q[ln(p(mu,pres))] - E_q[ln(q(mu,pres))]
if strcmp(op.cov_Type,'Full')    
    term4_1_all = zeros(1,K);
    invW0 = inv(prior.W0);
    for k=1:K
        mkMinusm0 = post.m(:,k) - prior.m0';%D*1
        term4_1_all(k) = - D*prior.beta0/post.beta(k) ...
            - prior.beta0*post.v(k)*(mkMinusm0'*post.W(:,:,k)*mkMinusm0) ...
            + (prior.v0-D-1)*extra.E_lnPresk(k) ...
            - post.v(k)*trace(invW0*post.W(:,:,k));
            %- post.v(k)*trace(post.W(:,:,k)\prior.W0);
    end
    term4_1 = 0.5*sum(term4_1_all);
    
    term4_2 = 0.5*D*sum(log(post.beta)) - sum(wishartEntropy(post.W,post.v));
    
    term4 = term4_1 - term4_2;    
elseif strcmp(op.cov_Type,'Diag')
    mkMinusm0 = post.m - repmat(prior.m0',1,K);%D*K
    term4_1 = - 0.5*prior.beta0*sum(sum(repmat(post.a,D,1) .* post.b .* mkMinusm0.^2)) ...
        + D*(prior.a0-1)*sum(extra.E_lnPresk) - prior.beta0*sum(sum(repmat(post.a,D,1).*post.b));
    term4_2 = sum(D*log(post.beta)) ...
        - sum(sum(repmat(post.a,D,1) - post.b + repmat(gammaln(post.a),D,1) + repmat((1-post.a).*psi(post.a),D,1)));
    term4 = term4_1 - term4_2;
elseif strcmp(op.cov_Type,'Fixed')    
    term4_1_all = zeros(1,K);
    for k=1:K
        mkMinusm0 = post.m(:,k) - prior.m0';%D*1
        term4_1_all(k) = - (mkMinusm0'*post.Pres(:,:,k)*mkMinusm0);
    end
    term4_1 = 0.5*sum(term4_1_all);
    
    term4_2 = 0;
    
    term4 = term4_1 - term4_2;    
end
freeEnergy = term1 + term2 + term3 + term4;

end
