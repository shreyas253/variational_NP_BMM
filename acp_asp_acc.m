function [acp asp acc]=acp_asp_acc(estim_classes,true_classes)

% acp_asp_acc calculates average cluster purities, average speaker purities, 
% and accuracies
%
% see: Ajmera, Bourland, Lapidot, McCowan: Unknown-multiple speaker clustering 
% using HMM, ICSLP, 2002.
% 
% implementation by ulpu

counts=crosstab(true_classes,estim_classes);
[num_true, num_estim]=size(counts);

N=sum(sum(counts));

for s=1:num_true
    sn(s)=sum(counts(s,:));
    sp(s)=sum(power(counts(s,:)/max(sn(s),1),2));
end

for c=1:num_estim
    cn(c)=sum(counts(:,c));
    cp(c)=sum(power(counts(:,c)/max(cn(c),1),2));
end

asp=(sp*sn')/N;
acp=(cp*cn')/N;

acc=sqrt(acp.*asp);