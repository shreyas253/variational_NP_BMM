% (C) 2016 Ulpu Remes, Shreyas Seshadri and Okko Rasaen
% MIT license
% For license terms and references, see README.txt
function model_1=train_variational_pitman_yor(data_1,T_1,max_kk,s_1,s_2,m_o,beta_o,a_o,b_o,rmx,thd1,init_method,sort_counts)

% data   observations (num observations x dimension)
% T      max num clusters

[N,D]=size(data_1);

% assert that observations have unit norm

assert(min(sum(power(data_1,2),2))>1-exp(-20));
assert(max(sum(power(data_1,2),2))<1+exp(-20));

% init model parameters

model_1.kk = repmat(a_o./b_o,1,T_1);
model_1.ln_kk = psi(a_o)-log(b_o);
model_1.kk_approx=model_1.kk-(a_o>1)*(1./b_o);

% note how s_1 and s_2 are mapped here!

g_2=s_1;
g_1=s_2;

% initialise responsibilities:

switch lower(init_method)

    case 'random'
        q_z=rand(N,T_1);
        q_z=bsxfun(@rdivide,q_z,sum(q_z,2));
        
    case 'kmeans'
        T_1=min(T_1,N);
        labels=kmeans(data_1,T_1,'distance','cosine','maxIter',rmx);
        q_z=dummyvar(labels);
        
        N_k=sum(q_z);
        [sorted_counts,sorted_index]=sort(N_k,'descend');
        q_z=q_z(:,sorted_index);
        
    otherwise
        error(['unknown init method: ' init])
end

N_k = sum(q_z); % E_q(q(z(n)=k))
C_k = sum(1-cumsum(q_z,2)); % E_q(q(z(n)>k)) 

% ELBO

const_kk = a_o.*log(b_o)-gammaln(a_o);

% do:

for rr=1:rmx
    
    % UPDATE VARIATIONAL MODEL PARAMETERS
    
    % 2.1. update dirichlet process param
    
    % mixture proportions/update variational distribution param:
    
     % g_1 = d;
     % g_1 = a_1
     
    gamma_1(1,:)=N_k-g_1+1;
    gamma_1(2,:)=C_k+T_1*g_1+g_2;
    
    % [E_q(ln_fii); E_q(ln_(1-fii))] expected stick proportions
    
    d_sum = psi(gamma_1(1,:)+gamma_1(2,:));
    E_pp=zeros(2,T_1);
    E_pp(1,:) = psi(gamma_1(1,:))-d_sum;
    E_pp(2,:) = psi(gamma_1(2,:))-d_sum;
    
    % E_q(ln_p(fii))+H_q(fii)
    % p: Beta(1,a)
    % q: Beta(g,g)
    
    E_p_H_p = gammaln((T_1-1)*g_1+g_2+1)-gammaln(1-g_1)-gammaln(g_2+T_1*g_1)+gammaln(gamma_1(1,:))+gammaln(gamma_1(2,:))-gammaln(gamma_1(1,:)+gamma_1(2,:)); % 1 x T
    E_p_H_p = E_p_H_p-(gamma_1(1,:)+g_1-1).*E_pp(1,:)-(gamma_1(2,:)-g_1-T_1*g_1).*E_pp(2,:);
    
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
    
    ln_q_z=bsxfun(@plus,E_px,E_pp(1,:)+cumsum(E_pp(2,:))-E_pp(2,:));
    
    % normalise
    
    q_z=exp(bsxfun(@minus,ln_q_z,max(ln_q_z,[],2))); 
    q_z=bsxfun(@rdivide,q_z,sum(q_z,2));
    
    assert(min(min(isfinite(q_z)))) % hope this is ok too
    
    N_k = sum(q_z); % E_q(q(z(n)=k))
    C_k = sum(1-cumsum(q_z,2)); % E_q(q(z(n)>k)) 
    
    % 1.3. update likelihood bound
    
    E_x=sum(q_z.*E_px)'; % 1 x T
    
    % E_q(ln_p(z))+H_q(z)
    
    E_z_H_z = N_k.*E_pp(1,:)+C_k.*E_pp(2,:)-sum(q_z.*log(q_z+exp(-700))); % 1 x T
    
    assert(isreal(E_z_H_z));
       
%     ELBO1(rr)=sum(E_x)/N;
%     ELBO2(rr)=sum(E_z_H_z)/N;
%     ELBO3(rr)=sum(E_m_H_m);
%     ELBO4(rr)=sum(E_k_H_k);
%     ELBO5(rr)=sum(E_p_H_p);
    
    ELBO(rr)=sum(E_x)+sum(E_z_H_z)+sum(E_p_H_p)+sum(E_m_H_m)+sum(E_k_H_k);
    
    if rr>1 && abs(ELBO(rr)-ELBO(rr-1))/N < thd1
       disp(ELBO(rr)/N)
       break 
    end
    
    % sort clusters
    
    if sort_counts
    
        [sorted_counts,sorted_index]=sort(N_k,'descend');
        q_z=q_z(:,sorted_index);
        N_k=sum(q_z);
        C_k = sum(1-cumsum(q_z,2));
        model_1.mu=model_1.mu(:,sorted_index);
        model_1.kk=model_1.kk(sorted_index);
        model_1.ln_kk=model_1.ln_kk(sorted_index);
        model_1.kk_approx=model_1.kk_approx(sorted_index);
    end
    
end % /iterations

% figure(1);
% plot(ELBO1);
% figure(2);
% plot(ELBO2);
% figure(3);
% plot(ELBO3);
% figure(4);
% plot(ELBO4);
% figure(5);
% plot(ELBO5);
% figure(6);
% plot(ELBO/N);

model_1.weight=N_k/N;
model_1.ELBO=ELBO;
model_1.q_z=q_z;
model_1.rr=rr;

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

