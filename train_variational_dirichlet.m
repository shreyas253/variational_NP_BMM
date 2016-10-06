function post=train_variational_dirichlet(data_1,prior,op)

% train_variational_dirichlet estimates a variational posterior distribution for DDVMFMM parameters
%
% data   observations (num observations x dimension)
%
% prior.mu   mean direction (1 x dimension)
% prior.beta   scale on concentration parameter
% prior.a   gamma distribution shape parameter
% prior.b   gamma distribution inverse scale parameter
% prior.alpha   concentration parameter
%
% op.K   num mixture components
% op.max_num_iter   max num iterations (proposed: 500)
% op.freetresh   iterations can be stopped when difference in ELBO < num observations x op.freetresh (proposed: 0.001)
% op.reorder   binary to indicate whether components are reordered (proposed: 0)
% op.init_Type   'random' allocate observations to components at random, 'kmeans' k-means initialisation, 'self' use op.init_z 
% op.init_z (optional)   initial allocations (1 x num components)
%
% output:
%
% post.mu   mean direction (dimension x num components)
% post.beta   scale on concentration parameter (1 x num components)
% post.a   gamma distribution shape parameter (1 x num components)
% post.b   gamma distribution inverse scale parameter (1 x num components)
% post.z   allocation probabilities (num observations x num components)
% post.fE   approximate free energy

[N,D]=size(data_1);

% assert that observations have unit norm

assert(min(sum(power(data_1,2),2))>1-exp(-20));
assert(max(sum(power(data_1,2),2))<1+exp(-20));

T_1=op.K; % num components

% mean direction:

m_o=repmat(prior.mu,T_1,1)'; % D x T
beta_o=prior.beta;

% concentration parameter:

a_o=prior.a;
b_o=prior.b;

model_1.kk = repmat(a_o./b_o,1,T_1);
model_1.ln_kk = psi(a_o)-log(b_o);
model_1.kk_approx=model_1.kk-(a_o>1)*(1./b_o);

max_kk=round(size(data_1,2)/2); 
while isfinite(besseli(size(data_1,2)/2,max_kk+10))
    max_kk=max_kk+10;
end

a_1=prior.alpha;

% initialise responsibilities:

switch lower(op.init_Type)

    case 'random'
        q_z=rand(N,T_1);
        q_z=bsxfun(@rdivide,q_z,sum(q_z,2));
        
    case 'kmeans'
        T_1=min(T_1,N);
        labels=kmeans(data_1,T_1,'distance','cosine','maxIter',op.max_num_iter);
        q_z=dummyvar(labels);
        
        N_k=sum(q_z);
        [sorted_counts,sorted_index]=sort(N_k,'descend');
        q_z=q_z(:,sorted_index);
        
    case 'self'
        q_z=dummyvar(op.init_z);
        N_k=sum(q_z);
        [sorted_counts,sorted_index]=sort(N_k,'descend');
        q_z=q_z(:,sorted_index);
    otherwise
        error(['unknown init method: ' init])
end

N_k = sum(q_z); % E_q(q(z(n)=k))

% ELBO

const_kk = a_o.*log(b_o)-gammaln(a_o);

% do:

for rr=1:op.max_num_iter
    
    % UPDATE VARIATIONAL MODEL PARAMETERS
    
    % 2.1. update dirichlet distribution param
    
    % mixture proportions/update variational distribution param:
    
    model_1.a=1+N_k;
     
    % [E_q(ln_fii); E_q(ln_(1-fii))] expected stick proportions
    
    E_pp = psi(model_1.a)-psi(sum(model_1.a));

    % E_q(ln_p(pi))+H_q(pi)
    % p: Beta(1,a)
    % q: Beta(g,g)
    
    % this is called "extra term" in kumatani's code (in case want to compare)
    
    E_p_H_p = gammaln(T_1*a_1)-T_1*gammaln(a_1)+(a_1-1)*sum(E_pp); % 1 x 1
    E_p_H_p = E_p_H_p+sum(gammaln(model_1.a))-gammaln(sum(model_1.a))-(T_1-sum(model_1.a))*psi(sum(model_1.a))-sum((model_1.a-1).*psi(model_1.a));
    
    assert(isreal(E_p_H_p));
    
    % 2.2. update observation distribution param

    % update mean distribution parameters:
    
    sum_x = data_1'*q_z; % D x T
    
    model_1.mu = bsxfun(@plus,sum_x,m_o*beta_o); % unnormalised mean
    model_1.beta = sqrt(sum(power(model_1.mu,2))); % 1 x T
    model_1.mu = bsxfun(@rdivide,model_1.mu, model_1.beta); % normalised mean
    
    % update concentration distribution parameters:
    
    nu=D/2-1;
    
    bes_kk = d_besseli(nu,model_1.kk_approx);
    bes_bo = d_besseli(nu,beta_o.*model_1.kk_approx);
    bes_bk = d_besseli(nu,model_1.beta.*model_1.kk_approx);
    
    model_1.a = a_o + N_k*nu; % positive 1 x T
    model_1.b = b_o + N_k.*bes_kk+beta_o.*bes_bo-model_1.beta.*bes_bk; % positive 1 x T
    
    %[max(bes_kk) max(bes_bo) max(bes_bk)]
    %save('this_crashed','model_1','bes_kk','bes_bo','bes_bk');

    assert(min(model_1.a)>0);
    assert(min(model_1.b)>0);
    
    % use previous parameter estimate as linearisation point (bessel function approximation is developed around this point):
    
    model_1.kk_approx=model_1.kk;
    model_1.kk_approx = min(model_1.kk_approx,max_kk); % must choose a point where besseli is finite
    
    % update expectation values:
    
    model_1.kk = model_1.a./model_1.b;
    model_1.ln_kk = psi(model_1.a)-log(model_1.b);
    
    assert(min(isfinite(model_1.kk))) % not sure this was ever an issue but does no harm to assert

    % calculate E_m_H_m, E_k_H_k
    
    % E_q(ln_p(mu))+H_q(mu)
    % p: F(m_o, beta_o*kk)
    % q: F(mu, beta*kk)
    
    % this is not used to optimise params so let us approximate besseli with a scaled exponential and avoid numerical issues:
           
    E_m_H_m = (nu+1/2)*log(beta_o./model_1.beta)+model_1.beta-beta_o+beta_o.*model_1.kk.*sum(m_o.*model_1.mu)-model_1.beta.*model_1.kk;
    
    % E_q(ln_p(kk))+H_q(kk)
    % p: G(a_o,b_o)
    % q: G(a,b)
    
    E_k_H_k = const_kk +(a_o-1).*model_1.ln_kk-b_o.*model_1.kk+model_1.a-log(model_1.b)-gammaln(model_1.a)+(1-model_1.a).*psi(model_1.a);


    % UPDATE RESPONSIBILITIES
    
    % 1.1. compute E_px, E_pp

    % E_q(ln_p(x|k)) expected observation likelihoods
    
    E_px = state_likelihood_bound(data_1,model_1);
    
    assert(isreal(E_px)) % this was an issue with one bessel approximation, hope is ok now
    
    % 1.2. compute expected state allocations
    
    ln_q_z=bsxfun(@plus,E_px,E_pp);
    
    % normalise
    
    q_z=exp(bsxfun(@minus,ln_q_z,max(ln_q_z,[],2))); 
    q_z=bsxfun(@rdivide,q_z,sum(q_z,2));
    
    assert(min(min(isfinite(q_z)))) % hope this is ok too
    
    N_k = sum(q_z); % E_q(q(z(n)=k))
    
    % 1.3. update likelihood bound
    
    E_x=sum(q_z.*E_px)'; % 1 x T
    
    % E_q(ln_p(z))+H_q(z)
    
    E_z_H_z = N_k.*E_pp-sum(q_z.*log(q_z+exp(-700))); % 1 x T
    
    assert(isreal(E_z_H_z));
    
    ELBO(rr)=sum(E_x)+sum(E_z_H_z)+sum(E_p_H_p)+sum(E_m_H_m)+sum(E_k_H_k);
    
    % check stopping condition
    
    if rr>1 && abs(ELBO(rr)-ELBO(rr-1))/N < op.freetresh
       disp(ELBO(rr)/N)
       break 
    end
    
    % sort clusters
    
    if op.reorder
    
        [sorted_counts,sorted_index]=sort(N_k,'descend');
        q_z=q_z(:,sorted_index);
        N_k=sum(q_z);
        model_1.mu=model_1.mu(:,sorted_index);
        model_1.kk=model_1.kk(sorted_index);
        model_1.ln_kk=model_1.ln_kk(sorted_index);
        model_1.kk_approx=model_1.kk_approx(sorted_index);
    end
    
end % /iterations

post.mu=model_1.mu;
post.beta=model_1.beta;
post.a=model_1.a;
post.b=model_1.b;
post.z=q_z;
post.fE=-1*ELBO(rr);

end

function bound = state_likelihood_bound(data1,model1)

    % calculate expected state likelihoods based on variational model params
    
    % have to approximate:
    
    bound=approximate_bound(data1,model1.mu,model1.kk,model1.ln_kk,model1.kk_approx);
end

function bound=approximate_bound(data,mu,kk,ln_kk,kk_approx)

    % calculate lower bound on the expected state likelihoods
    %
    % data N x D
    % mu D x T
    % kk 1 x T E_q(kk)
    % ln_kk 1 x T E_q(ln_kk)
    % kk_approx 1 x T 
    %
    % return: bound
    
    nu=size(mu,1)/2-1;
    
    % calculate upper bound on the bessel term:
    
    bes_bound=log(besseli(nu,kk_approx)+exp(-700))+d_besseli(nu,kk_approx).*(kk-kk_approx);
    
    % calculate likelihood bound:
    
    c_k = nu*ln_kk-(nu+1)*log(2*pi)-bes_bound; % normalisation term
    
    distances = data*mu; % N x T
    bound = bsxfun(@plus,bsxfun(@times,distances,kk),c_k);   
end

function bes = d_besseli(nu,kk)

    % kk here can exceed max_kk so need to approximate sometimes
    
    try
        bes=besseli(nu+1,kk)./(besseli(nu,kk)+exp(-700))+nu./kk;
        assert(min(isfinite(bes)))
    catch me
        bes=sqrt(1+power(nu,2)./power(kk,2));
    end
end

