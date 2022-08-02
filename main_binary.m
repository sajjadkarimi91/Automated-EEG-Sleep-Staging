close all
clc
clear
addpath(genpath(pwd))

dataset_dir = 'D:/PHD codes/DataSets/sleepedf20/';


%% load data & preprocessing

sub = 5; % total subjects

dm_common_load;

Fs = 100; % sampling rate


epoch = 30;
len = epoch*3; % each feature is extracted from 90s EEG signal;
boundary = 1;
% 1: avoiding edge effects by considering a much longer signal temporarily
% 0: the edge artifacts may contaminate the part of the feature we are interested in.

num_signal = min(2*sub,39)*2;

Channels = cell((num_signal)/2, 3);
label_org = cell((num_signal)/2, 1);


for caseNo = 1:min(2*sub,39)

    sub_id = ceil(caseNo/2);
    d_id = 1+(1-rem(caseNo,2));

    PSG1 = all_record{sub_id,d_id}(:,1);
    PSG2 = all_record{sub_id,d_id}(:,2);
    EOG = all_record{sub_id,d_id}(:,3);
    STAGE = all_hypnogram{sub_id,d_id};

    before = (2+boundary);
    after =  boundary;
    [x1,x2,num_STAGE,~] = truncated_rawPSG_HYP_SC(STAGE,PSG1,PSG2,(240+before),(60+after)); % include only 120 mins before and after sleep
    [~,x3,num_STAGE,~] = truncated_rawPSG_HYP_SC(STAGE,PSG1,EOG,(240+before),(60+after)); % include only 60 mins before and after sleep


    RR = rem(length(x1),Fs*epoch);
    x1 = x1(1:length(x1)-RR);

    RR = rem(length(x2),Fs*epoch);
    x2 = x2(1:length(x2)-RR);

    RR = rem(length(x3),Fs*epoch);
    x3 = x3(1:length(x3)-RR);

    num_STAGE(num_STAGE>1) = 2; % wake condition
    label_org{caseNo} = num_STAGE;

    Channels{caseNo, 1} = x1; % FPZCZ
    Channels{caseNo, 2} = x2; % PZOZ
    Channels{caseNo, 3} = x3; % EOG

    if (length(x1)/100/30~=length(num_STAGE))
        error('error');
    end
    if (length(x2)/100/30~=length(num_STAGE))
        error('error');
    end

end
clear x1 x2 PSG1 PSG2 num_STAGE


subject_infos.PID = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,...
    11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20]'; %
subject_infos.PID = subject_infos.PID(1:min(sub*2,39));

%% stacking the signals for the scattering transform

windowSize = 3000;
for i = 1:size(Channels, 1)
    i
    [~, vec_eeg_features] = eeg_features_channel(Channels{i, 1} , windowSize , Fs);  % FPZCZ
    Channels{i, 1} = vec_eeg_features(:,4:end-1);

    [~, vec_eeg_features] = eeg_features_channel(Channels{i, 2} , windowSize , Fs);% PZOZ
    Channels{i, 2} = vec_eeg_features(:,4:end-1);

    [~, vec_eeg_features] = eeg_features_channel(Channels{i, 3} , windowSize , Fs);% EOG
    Channels{i, 3} = vec_eeg_features(:,4:end-1);

    label{i} = label_org{i}(4:end-1);
    if (size(Channels{i, 1},2)~=length(label{i}))
        error('error in label alignment');
    end
end

Channels_org = Channels;

save('channels_b.mat','Channels_org','label')

%% organization, labelling and indexing

Channels = Channels_org ;

true_labels = cell(size(Channels, 1), 1);
subjectId = cell(size(subject_infos.PID));
for i = 1:size(Channels, 1)
    id = find(label{i} > 0);
    if length(id)<length(label{i})
        disp('remove')
    end
    for j = 1:size(Channels, 2)
        Channels{i, j} = Channels{i, j}(:,id); % removing the features corresponding to MOVEMENT
    end
    true_labels{i} = full(ind2vec(double(label{i}(id)')))';
    subjectId{i} = repelem(subject_infos.PID(i), length(true_labels{i}))';
end


[X,Y,Z] = my_cell2mat3(Channels,1);

% concatenation
feature_sets{1,1} = X';
feature_sets{2,1} = Y';
feature_sets{3,1} = Z';

save('fe_channels_b.mat','feature_sets','true_labels','subjectId')


%% dimension reduction & visulization

feature_sets_mat = [feature_sets{1,1},feature_sets{2,1},feature_sets{3,1}];
feature_sets_mat = zscore(feature_sets_mat);
dim = 2;
num_classes = 5;

train_data = feature_sets_mat;
[~, train_label] = max(cell2mat(true_labels), [], 2);

[para_lda, Z_lda] = lda_sldr(train_data, train_label, dim); % Linear discriminant analysis (LDA)
% [para_hlda, Z_hlda] = plsda_sldr(train_data, train_label, dim); % Heteroscedastic extension of LDA
[coeff, Z_hlda, ~, ~, ~, mu] = pca(train_data,'NumComponents',dim);

sz = 5;
figure
subplot(2,1,1)
gscatter(Z_hlda(:,1),Z_hlda(:,2),train_label)
title('PCA')
grid on

subplot(2,1,2)
histogram(Z_lda(train_label==1,1))
hold on
histogram(Z_lda(train_label==2,1))
title('LDA')
grid on


figure
tsne_features = tsne(feature_sets_mat);
gscatter(tsne_features(:,1),tsne_features(:,2),train_label);
grid on

save('tsne_features_b.mat','tsne_features','true_labels','subjectId')

%% 5-fold cross validation

y = cell2mat(true_labels);
[~, response] = max(y, [], 2);

cv_groups = cvpartition(response,'KFold',5);

temp_train = cv_groups.training(1);
temp_test = cv_groups.test(1);

%% leave-one-out cross validation

y = cell2mat(true_labels);
[~, response] = max(y, [], 2);
% create leave-one-out cross validation partition
cvp = struct;
cvp.NumObservations = size(response, 1);
cvp.testSize = zeros(1, sub);
cvp.trainSize = zeros(1, sub);
cvp.testStore = cell(1, sub);
cvp.trainStore = cell(1, sub);

for i = 1:sub
    cvp.testStore{i} = cell2mat(subjectId) == i;
    cvp.testSize(i) = sum(cvp.testStore{i});
    cvp.trainSize(i) = cvp.NumObservations - cvp.testSize(i);
    cvp.trainStore{i} = ~cvp.testStore{i};
end

%% 5-class SVM One-vs-One
close all
svm_flag = 1;

feature_sets_mat = [feature_sets{1,1},feature_sets{2,1},feature_sets{3,1}];
feature_sets_mat = zscore(feature_sets_mat);
predictors = feature_sets_mat;

prediction = cell(sub, 1);
test_labels= cell(sub, 1);
prediction_score= cell(sub, 1);

for i = 1:sub
    disp(['fold: ',num2str(i)])

    predictors_this = predictors;
    num_features = size(predictors_this,2);

    if svm_flag==0
        template = templateKNN('NumNeighbors',20,'Standardize',1);
    else
        template = templateSVM('KernelFunction', 'linear', ...
            'PolynomialOrder', [], 'KernelScale', [], ...
            'BoxConstraint', 0.3, 'Standardize', true);
    end
    Mdl = fitcecoc(predictors_this(cvp.trainStore{i}, :), response(cvp.trainStore{i}, :), ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', [1; 2; 3; 4; 5]);

    [~, validationScores] = predict(Mdl, predictors_this(cvp.testStore{i}, :));
    [~, prediction{i}] = max(validationScores, [], 2);
    test_labels{i} = response(cvp.testStore{i});
    prediction_score{i} = validationScores(:,2);
end

acc = sum(cell2mat(test_labels)==cell2mat(prediction))/length(cell2mat(prediction));
disp(['Accuracy is: ',num2str(100*acc)])

%plot ROC & PRC
test_labels_mat = cell2mat(test_labels);
test_labels_mat(test_labels_mat>1) = 2;
test_labels_mat = test_labels_mat-1;
prediction_score_mat = cell2mat(prediction_score);
prec_rec(prediction_score_mat,test_labels_mat ,'numThresh',length(prediction_score_mat));

%plot confusion matrix
output_label_mat = cell2mat(prediction);
plotconfusion(test_labels_mat,output_label_mat);

%% lda & plsda & 5-class SVM One-vs-One with
close all
%common settings
svm_flag = 0;
dim = 4;
lda_flag = 0;

y = cell2mat(true_labels);
[~, response] = max(y, [], 2);

feature_sets_mat = [feature_sets{1,1},feature_sets{2,1},feature_sets{3,1}];
feature_sets_mat = zscore(feature_sets_mat);

predictors = feature_sets_mat;

prediction = cell(sub, 1);
test_labels= cell(sub, 1);
prediction_score= cell(sub, 1);

for i = 1:sub
    disp(['fold: ',num2str(i)])

    train_data = predictors(cvp.trainStore{i},:);
    train_label = response(cvp.trainStore{i});

    if lda_flag==1
        [para_lda, Z_lda] = lda_sldr(train_data, train_label, dim); % Linear discriminant analysis (LDA)
        predictors_this = test_sldr(predictors, para_lda);
    else
        [coeff, predictors_this, ~, ~, ~, mu] = pca(predictors,'NumComponents',dim);
    end

    num_features = size(predictors_this,2);
    if svm_flag==0
        template = templateKNN('NumNeighbors',20,'Standardize',1);
    else
        template = templateSVM('KernelFunction', 'linear', ...
            'PolynomialOrder', [], 'KernelScale', [], ...
            'BoxConstraint', 0.3, 'Standardize', true);
    end

    Mdl = fitcecoc(predictors_this(cvp.trainStore{i}, :), response(cvp.trainStore{i}, :), ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', [1; 2; 3; 4; 5]);

    [~, validationScores] = predict(Mdl, predictors_this(cvp.testStore{i}, :));
    [~, prediction{i}] = max(validationScores, [], 2);
    test_labels{i} = response(cvp.testStore{i});
    prediction_score{i} = validationScores(:,2);
end

acc = sum(cell2mat(test_labels)==cell2mat(prediction))/length(cell2mat(prediction));
disp(['Accuracy is: ',num2str(100*acc)])

%plot ROC & PRC
test_labels_mat = cell2mat(test_labels);
test_labels_mat(test_labels_mat>1) =2;
test_labels_mat = test_labels_mat-1;
prediction_score_lda = cell2mat(prediction_score);
prec_rec(prediction_score_lda,test_labels_mat ,'numThresh',length(prediction_score_lda));

%% TSNE & 5-class SVM One-vs-One
close all
%common settings
svm_flag = 0;

predictors = tsne_features;

prediction = cell(sub, 1);
test_labels= cell(sub, 1);
prediction_score= cell(sub, 1);

for i = 1:sub
    disp(['fold: ',num2str(i)])

    predictors_this = predictors;
    num_features = size(predictors_this,2);

    if svm_flag==0
        template = templateKNN('NumNeighbors',20,'Standardize',1);
    else
        template = templateSVM('KernelFunction', 'linear', ...
            'PolynomialOrder', [], 'KernelScale', [], ...
            'BoxConstraint', 0.3, 'Standardize', true);
    end

    Mdl = fitcecoc(predictors_this(cvp.trainStore{i}, :), response(cvp.trainStore{i}, :), ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', [1; 2; 3; 4; 5]);

    [~, validationScores] = predict(Mdl, predictors_this(cvp.testStore{i}, :));
    [~, prediction{i}] = max(validationScores, [], 2);
    test_labels{i} = response(cvp.testStore{i});
    prediction_score{i} = validationScores(:,2);
end

acc = sum(cell2mat(test_labels)==cell2mat(prediction))/length(cell2mat(prediction));
disp(['Accuracy is: ',num2str(100*acc)])

%plot ROC & PRC
test_labels_mat = cell2mat(test_labels);
test_labels_mat(test_labels_mat>1) = 0;
test_labels_mat = test_labels_mat-1;
prediction_score_tsne = cell2mat(prediction_score);
prec_rec(prediction_score_tsne,test_labels_mat ,'numThresh',length(prediction_score_tsne));

