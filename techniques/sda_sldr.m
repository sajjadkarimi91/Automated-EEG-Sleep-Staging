function [para, Z] = sda_sldr(X, labels, dim, lamda_in, eps_val_in)

% Stochastic discriminant analysis (SDA) matches similarities between points in the projection space with those in a response space
% Juuti, Mika, Francesco Corona, and Juha Karhunen.
% Stochastic discriminant analysis for linear supervised dimension reduction.
% Neurocomputing 291 (2018): 136-150.

%[para,W] = sda_sldr(X, labels, dim) , where dim values by default is Number of classes: C
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    labels: n --- dimensional vector of class labels
%    dim:    ----- dimensionality of reduced space (default:C)
%            dim has to be from 1<=dim<=d
% Output:
%    para:   output structure of sda model for input of test_sldr.m function
%    Z:      n x dim matrix of dimensionality reduced features

global dummy_labels X_org count_itr lamda P_sigma

classes_labels = unique(labels);
num_classes = length(classes_labels);

[n,d] = size(X);

if(nargin==2)
    dim= min(num_classes,max(1,n-1));
end

if nargin<4
    lamda = 0.01;
else
    lamda = lamda_in;
end

if nargin<5
    eps_val = 10^-5;
else
    eps_val = eps_val_in;
end



count_itr=0;

dummy_labels = eps_val + zeros(size(X,1),num_classes);
for k = 1:num_classes
    dummy_labels(labels ==classes_labels(k),k) = 1;
end

P_sigma = 0;
for i = 1:size(dummy_labels,1)
   
    [~,ind_class] = max(dummy_labels(i,:));
    Pi = dummy_labels(:,ind_class);
    P_sigma = P_sigma + sum(Pi);
end


% recentering original feature
mb = mean(X,'omitnan');
X = X - mb;
X_org = X;

%Initialize W using PCA
coeff = pca(X,'NumComponents',dim);
% Z = (yhat-mu)*coeff;
W = coeff;
x0 = W(:); % vectorize matrix

options = optimoptions('fminunc','MaxFunctionEvaluations',5*10^2,'Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'Display','iter','MaxIterations',20);
% options = optimoptions('fmincon','MaxFunctionEvaluations',5*10^4,'SpecifyObjectiveGradient',true,'StepTolerance',10^-8,'ConstraintTolerance',10^-8);

W = fminunc(@sda_cost,x0,options);
W = reshape(W,d,[]);
% Orthogonalize W t using thin singular value decomposition
[U,S,V] = svd(W);
W =  U*S;


% Z has the dimentional reduced data sample X.
Z = X*W;

para.W = W;
para.mb = mb;
para.model = 'sda';

end

%% cost function and its total gradient
% For large-scale data sets, we try to write memory-efficient code instead of fast code


function [f, g] = sda_cost(w_vec)

global dummy_labels X_org count_itr lamda P_sigma

count_itr = count_itr+1;

f = 0;
g=0;

f = f + lamda*sum(w_vec.^2);
g = g + 2*lamda*w_vec;

[K,d] = size(X_org);
W = reshape(w_vec,d,[]);
Z = X_org*W;
dim = length(w_vec)/d;

Q_sigma = 0;
for i = 1:K
    D2i = sum((Z(i,:)-Z).^2,2);
    Qbi = 1./(1+D2i);
    Q_sigma = Q_sigma + sum(Qbi);
end

for i = 1:K

    D2i = sum((Z(i,:)-Z).^2,2);
    Qi = (1/Q_sigma)./(1+D2i);
    Qbi = Q_sigma*Qi;

    [~,ind_class] = max(dummy_labels(i,:));
    Pi = dummy_labels(:,ind_class)/P_sigma;

    Ti = X_org(i,:) - X_org;
    
    g1 = squeeze(sum(repmat(Ti,1,1,dim).*permute(repmat((Ti*W).*repmat((Pi-Qi).*Qbi,1,dim),1,1,d),[1,3,2]),1));

    g = g + g1(:);
    f = f + Pi' * log(Pi./(Qi+10^-20));
end


end

