clc
clear
close all
addpath(genpath(pwd))
rng(1);

% this run is a heteroscedastic scenario example
%%
N_init = 2000; % sample per each class
dec_rate= 1;
d = 20; % dimensionality of original features
num_classes = 4;
dim = 2; % dimensionality of reduced space
similar_cov = 0; % (0->heteroscedastic), and (1->homoscedastic) covariance matrices
separation_factor = 0.2; % (0.01<val< 0.5) Separation of classes is controlled by this parameter

%% parameter init

for k=1:num_classes
    N(k)= round(N_init*dec_rate^k);
    class_means(:,k) = separation_factor*randn(d,1)+k*separation_factor/3;
    if k==1
        A{k} = (0.1+rand(d,d))/sqrt(d);
    else
        if similar_cov==1
            A{k} = A{1};
        else
            temp = (0.1+rand(d,d))/sqrt(d);
            ind_zero = randperm(length(temp(:)));
            temp(ind_zero(1:floor(d^2/2)))=0;
            A{k} = rand(d,d)/sqrt(d);
        end
    end
end


%% data generation

train_data = zeros(sum(N),d);
train_label = zeros(sum(N),1);
cum_N = [0,cumsum(N)];
for k=1:num_classes
    train_data(cum_N(k)+1:cum_N(k+1),:)  = (0.2+rand(1))*((randn(N(k),d)*A{k}) + class_means(:,k)');
    train_label(cum_N(k)+1:cum_N(k+1))=k;
end

%% dimension reduction


[para_lda, Z_lda] = lda_sldr(train_data, train_label, dim); % Linear discriminant analysis (LDA)

[para_hlda, Z_hlda] = hlda_sldr(train_data, train_label, dim); % Heteroscedastic extension of LDA
try
    [para_mmda, Z_mmda] = mmda_sldr(train_data, train_label, dim); % Max-min distance analysis (MMDA)
    [para_mmda, Z_whmmda] = whmmda_sldr(train_data, train_label, dim); % Heteroscedastic extension of MMDA
catch
    warning('please add cvx for MMDA')
    Z_mmda = Z_lda;
    Z_whmmda = Z_hlda;
    warning('MMDA was replaced with LDA to continue this example')
    warning('WHMMDA was replaced with HLDA to continue this example')
end

[para_plsda, Z_plsda] = plsda_sldr(train_data, train_label, dim);% Partial least squares discriminant analysis (PLS‐DA)

[para_sda, Z_sda] = sda_sldr(train_data, train_label, dim); % Stochastic discriminant analysis (SDA)

%% some EDA to analysis the results

sz = 5;
figure
subplot(6,1,1)
scatter(Z_sda(:,1),Z_sda(:,2),sz,train_label/num_classes,'filled')
title('SDA')
grid on

subplot(6,1,2)
scatter(Z_whmmda(:,1),Z_whmmda(:,2),sz,train_label/num_classes,'filled')
title('WHMMDA')
grid on

subplot(6,1,3)
scatter(Z_mmda(:,1),Z_mmda(:,2),sz,train_label/num_classes,'filled')
title('MMDA')
grid on

subplot(6,1,4)
scatter(Z_hlda(:,1),Z_hlda(:,2),sz,train_label/num_classes,'filled')
title('HLDA')
grid on

subplot(6,1,5)
scatter(Z_lda(:,1),Z_lda(:,2),sz,train_label/num_classes,'filled')
title('LDA')
grid on

subplot(6,1,6)
scatter(Z_plsda(:,1),Z_plsda(:,2),sz,train_label/num_classes,'filled')
title('PLSDA')
grid on



