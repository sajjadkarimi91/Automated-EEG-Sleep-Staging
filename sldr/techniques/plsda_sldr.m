function [para, Z] = plsda_sldr(X, labels, dim)

% Partial least squares (PLS) has a long history & PLS constructs the basis of a linear subspace iteratively
% This function considers the multi‐class partial least squares discriminant analysis (PLS‐DA) and
% the unsupervised objective in PCA

% Pomerantsev, Alexey L., and Oxana Ye Rodionova.
% "Multiclass partial least squares discriminant analysis: Taking the right way—A critical tutorial."
% Journal of Chemometrics 32, no. 8 (2018): e3030.

%[para,W] = plsda_sldr(X, labels, dim) , where dim values by default is
%Number of classes: C-1
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    labels: n --- dimensional vector of class labels
%    dim:    ----- dimensionality of reduced space (default:C-1)
%            dim has to be from 1<=dim<=C-1
% Output:
%    para:   output structure of PLS‐DA model for input of test_sldr.m function
%    Z:      (n x dim) matrix of dimensionality reduced features

[n ,d]= size(X);
classes_labels = unique(labels);
num_classes = length(classes_labels);

dummy_labels = zeros(n,num_classes);

for k = 1:num_classes
    dummy_labels(labels ==classes_labels(k),k) = 1;
end

if(nargin==2)
    dim = num_classes-1;
end

if dim>=num_classes
    dim = num_classes-1;
    warning('dim was set to C-1')
end

% recentering original feature
mb = mean(X,'omitnan');
X = X - mb;

ncomp = min(3*dim,d);
[~,~,~,~,beta_coef] = plsregress(X,dummy_labels,ncomp);

yhat = [ones(n,1) X]*beta_coef;

[coeff, score, ~, ~, ~, mu] = pca(yhat,'NumComponents',dim);

Z = score;
% Z = (yhat-mu)*coeff;

para.mu_pca = mu;
para.W = coeff;
para.mb = mb;
para.beta_coef = beta_coef;
para.model = 'plsda';

end