% (C) 2016 Shreyas Seshadri, Ulpu Remes and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function [ post,extra ] = postUpdate(x,r,prior,op)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

K = size(r, 2);
[N,D] = size(x);

threshold_for_Nk = 1.0e-200; % to avoid the problem of infinity
Nk = sum(r,1); % 1*K
I = find(Nk>threshold_for_Nk);
inv_Nk = zeros(1,K);
inv_Nk(I) = 1./Nk(I);

xkBarProdNk = x' * r; % D*K
xkBar = xkBarProdNk .* repmat(inv_Nk, D, 1); % D*K 
extra.xkBar = xkBar;

if strcmp(op.cov_Type,'Full')
    post.v = prior.v0 + Nk; %1*K
    post.beta = prior.beta0 + Nk; %1*K
    post.W = zeros(D,D,K); % D*D*K
    post.m = (xkBarProdNk + repmat(prior.beta0*prior.m0',1,K)) ...
        ./ repmat(post.beta,D,1); %D*K
    invW0 = inv(prior.W0);
    for k=1:K
      xMinusxkBar = x - repmat(xkBar(:,k)',N,1); % N*D
      SkProdNk = (repmat(r(:,k),1,D).*xMinusxkBar)'*xMinusxkBar; %D*D
      xkBarMinusm0 = xkBar(:,k) - prior.m0'; %D*1
      post.invW{k} =  invW0 + SkProdNk ...
          + Nk(k)*prior.beta0*(xkBarMinusm0*xkBarMinusm0')/(post.beta(k)); % D*D
      post.W(:,:,k) = inv(post.invW{k});
      extra.SkProdNk{k} = SkProdNk;
    end
       
elseif strcmp(op.cov_Type,'Diag') 
    post.a = prior.a0 + (Nk/2); %1*K (not D*K because we have set the same prior for each dimension and the posterior term (Nk/2) is constant for each dim)
    post.beta = prior.beta0 + Nk; %1*K
    post.m = (xkBarProdNk + repmat(prior.beta0*prior.m0',1,K)) ...
        ./ repmat(post.beta,D,1); %D*K
    post.b = zeros(D,K); % D*K
    for k=1:K
      xMinusxkBar = x - repmat(xkBar(:,k)',N,1); % N*D
      SkProdNk = 0.5*sum(repmat(r(:,k),1,D).*(xMinusxkBar.^2),1)'; %D*1
      xkBarMinusm0 = xkBar(:,k) - prior.m0';%D*1
      post.b(:,k) = 1./prior.b0 + SkProdNk ...
          + Nk(k)*prior.beta0*(xkBarMinusm0.^2)/(post.beta(k))/2; % D*1
     extra.SkProdNk{k} = SkProdNk;
     end
     post.b = 1./post.b;

elseif strcmp(op.cov_Type,'Fixed')
    mTmp = (op.PresMain*xkBarProdNk + prior.Pres0*repmat(prior.m0',1,K)); %D*K
    for k=1:K
        xMinusxkBar = x - repmat(xkBar(:,k)',N,1); % N*D
        extra.SkProdNk{k} = (repmat(r(:,k),1,D).*xMinusxkBar)'*xMinusxkBar; %D*D
        post.m(:,k) = inv(prior.Pres0+(Nk(k)*op.PresMain)) * mTmp(:,k); %D*K        
        post.Pres(:,:,k) = prior.Pres0+ Nk(k)*op.PresMain;
    end
end
if strcmp(op.Pi_Type,'DP')
    post.gamma = zeros(2,K-1);
    post.gamma(1,:) = 1 + Nk(1:K-1);
    post.gamma(2,:) = prior.alpha + sum(Nk) - cumsum(Nk(1:K-1),2);
elseif strcmp(op.Pi_Type,'DD')
    post.DDalpha = ones(size(Nk))*prior.alpha + Nk;
elseif strcmp(op.Pi_Type,'PYP')
    post.gamma = zeros(2,K-1);
    post.gamma(1,:) = 1 - prior.g + Nk(1:K-1);
    post.gamma(2,:) = prior.alpha + (1:K-1)*prior.g + sum(Nk) - cumsum(Nk(1:K-1),2);
end

end

