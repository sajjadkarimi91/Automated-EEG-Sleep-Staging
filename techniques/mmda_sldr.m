function [para, Z] = mmda_sldr(X, labels, dim)

% download CVX toolbox from below link:
% http://web.cvxr.com/cvx/cvx-w64.zip

% MMDA method for supervised linear dimension reduction (LDR).
% Bian, Wei, and Dacheng Tao.
% "Max-min distance analysis by using sequential SDP relaxation for dimension reduction."
% IEEE Transactions on Pattern Analysis and Machine Intelligence 33, no. 5 (2010): 1037-1050.

%[para,W] = mmda_sldr(X, labels, dim) , where dim values by default is Number of classes: C
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    labels: n --- dimensional vector of class labels
%    dim:    ----- dimensionality of reduced space (default:C)
%            dim has to be from 1<=dim<=d
% Output:
%    para:   output structure of mmda model for input of test_sldr.m function
%    Z:      n x dim matrix of dimensionality reduced features

d = size(X,2);
classes_labels = unique(labels);
num_classes = length(classes_labels);

if(nargin==2)
    dim= min(num_classes,max(1,size(X,2)-1));
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


Sw_sqrt = (Sw)^0.5;
Sw_sqrtinv = (real(inv(Sw_sqrt))+real(inv(Sw_sqrt))')/2;

% apply the whitening preprocessing
% X = (Sw_sqrtinv * X' )'; % both are equivalent
X_data = X * Sw_sqrtinv ;
X_org = X;

for k = 1:num_classes
    M(k,:) = mean(X_data(labels==classes_labels(k),:),'omitnan');
end

for i=1:num_classes
    for j=i+1:num_classes
        m_i = M(i,:)';
        m_j = M(j,:)';
        Dij = (m_i-m_j)*(m_i-m_j)';
        Aij = Dij/(p(i)*p(j)+10^-10);
        A{i,j} = Aij;
    end
end



%###############################
% global SDP relaxation of MMDA
%###############################

I = eye(d);
%=======================================================
cvx_begin sdp

variable X(d,d) hermitian
variable t
minimize -t
subject to
for i = 1:num_classes
    for j = i+1:num_classes
        trace(A{i,j}*X) >= t
    end
end
trace(X) == dim
X >= 0
I-X>=0 ;

cvx_end
%=========================================================
X_opt = X;


[V,D] = eig(X_opt);
D = real(diag(D));
V = real(V);
[~,sort_index]=sort(D,'descend');
W_app =  V(:,sort_index(1:dim));

%###############################
% Iterative Local SDP Relaxation
%###############################

X0 = W_app*W_app';
etha_vals = 10.^(-linspace(2,6,20));

for itr = 1:20

    %=======================================================
    cvx_begin sdp

    variable X(d,d) symmetric
    variable t
    minimize -t
    subject to
    for i = 1:num_classes
        for j = i+1:num_classes
            trace(A{i,j}*X) >= t
        end
    end
    trace(X) == dim
    X >= 0
    I-X>=0
    trace(inv(X0+I)*X) <= (1+etha_vals(itr))*trace(inv(X0+I)*X0)
    trace(inv(X0+I)*X) >= (1-etha_vals(itr))*trace(inv(X0+I)*X0);

    cvx_end
    %=========================================================

    X_opt = real(X);

    % selecting dim eigenvectors associated with the dim largest eigenvalues
    [V,D] = eig(X_opt);
    D = real(diag(D));
    V = real(V);
    [~,sort_index]=sort(D,'descend');
    W_app =  V(:,sort_index(1:dim));

    % Iterative Local SDP Relaxation
    X0 = W_app*W_app';


end

W = Sw_sqrtinv * W_app;

% Z has the dimentional reduced data sample X.
Z = X_org*W;

para.W = W;
para.mb = mb;
para.model = 'mmda';

end