function [para, Z] = lda_sldr(X, labels, dim)

% Linear discriminant analysis (LDA) is probably the most well-known approach to supervised linear
% dimension reduction (LDR).
% This classical technique was developed by Fisher

% R.A. Fisher, “The Use of Multiple Measurements in Taxonomic
% Problems,” Annals of Eugenics, vol. 7, pp. 179-188, 1936.

% C.R. Rao, “The Utilization of Multiple Measurements in Problems
% of Biological Classification,” J. Royal Statistical Soc., Series B, vol. 10,
% pp. 159-203, 1948.

%[para,W] = lda_sldr(X, labels, dim) , where dim values by default is Number of
%classes: C-1
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    labels: n --- dimensional vector of class labels
%    dim:    ----- dimensionality of reduced space (default:C)
%            dim has to be from 1<=dim<=C-1
% Output:
%    para:   output structure of lda model for input of test_sldr.m function
%    Z:      n x dim matrix of dimensionality reduced features


classes_labels = unique(labels);
num_classes = length(classes_labels);

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

Sw = 0;
SB = 0;
for k = 1:num_classes

    Si{k}= cov( X(labels==classes_labels(k),:) ,1, 'omitrows' );
    M(k,:) = mean(X(labels==classes_labels(k),:),'omitnan');
    p(k) = sum(labels==classes_labels(k))/length(labels);
    Sw = Sw + p(k)*Si{k};
    SB = SB + p(k)*M(k,:)' * M(k,:);

end


% selecting dim eigenvectors associated with the dim largest eigenvalues
% [V,D] = eig(inv(Sw)*Sb);
% D = real(diag(D));
% V = real(V);
% [~,sort_index]=sort(D,'descend');
% W =  V(:,sort_index(1:dim));


% Perform eigendecomposition of inv(Sw)*Sb
% SB = cov(X ,1, 'omitrows' )-Sw;
[M1, lambda] = eig(SB, Sw);
[~, ind] = sort(diag(lambda), 'descend');
W = M1(:,ind(1:min([dim size(M1, 2)])));


% Z has the dimentional reduced data sample X.
Z = X*W;

para.W = W;
para.mb = mb;
para.model = 'lda';
end