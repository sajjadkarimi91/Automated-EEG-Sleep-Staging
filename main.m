close all
clc
clear
addpath(genpath(pwd))

dataset_dir = 'D:/PHD codes/DataSets/sleepedf20/';


%% load data & preprocessing

dm_common_load;

Fs = 100; % sampling rate

sub = 20; % total subjects
epoch = 30;
len = epoch*3; % each feature is extracted from 90s EEG signal;
boundary = 1;
% 1: avoiding edge effects by considering a much longer signal temporarily
% 0: the edge artifacts may contaminate the part of the feature we are interested in.

num_signal = min(2*sub,39)*2;

Channels = cell((num_signal)/2, 3);
label = cell((num_signal)/2, 1);


for caseNo = 1:min(2*sub,39)

    sub_id = ceil(caseNo/2);
    d_id = 1+(1-rem(caseNo,2));

    PSG1 = all_record{sub_id,d_id}(:,1);
    PSG2 = all_record{sub_id,d_id}(:,2);
    EOG = all_record{sub_id,d_id}(:,3);
    STAGE = all_hypnogram{sub_id,d_id};

    before = (2+boundary);
    after =  boundary;
    [x1,x2,num_STAGE,~] = truncated_rawPSG_HYP_SC(STAGE,PSG1,PSG2,(60+before),(60+after)); % include only 30 mins before and after sleep
    [~,x3,num_STAGE,~] = truncated_rawPSG_HYP_SC(STAGE,PSG1,EOG,(60+before),(60+after)); % include only 30 mins before and after sleep


    RR = rem(length(x1),Fs*epoch);
    x1 = x1(1:length(x1)-RR);

    RR = rem(length(x2),Fs*epoch);
    x2 = x2(1:length(x2)-RR);

    RR = rem(length(x3),Fs*epoch);
    x3 = x3(1:length(x3)-RR);

    label{caseNo} = num_STAGE;

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

    label{i} = label{i}(4:end-1);
    if (size(Channels{i, 1},2)~=length(label{i}))
        error('error in label alignment');
    end
end

Channels_org = Channels;

save('channels.mat','Channels_org')

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

save('fe_channels.mat','feature_sets')

%%




%% 5-class SVM One-vs-One

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


for ch = 1:3

    predictors = feature_sets{ch,1};

    template = templateSVM('KernelFunction', 'gaussian', ...
        'PolynomialOrder', [], 'KernelScale', 10, ...
        'BoxConstraint', 1, 'Standardize', true);

    template = templateSVM('KernelFunction', 'linear', ...
        'PolynomialOrder', [], 'KernelScale', [], ...
        'BoxConstraint', 1, 'Standardize', true);

    prediction = cell(sub, 1);
    for i = 1:sub
        disp([ch,i])

        ranking_ch = [];
        for c1 = 1:5
            for c2 = c1+1:5

                temp_resp = response;
                ind_train = (temp_resp == c1 | temp_resp == c2)&cvp.trainStore{i};
                y_train = temp_resp(ind_train);
                y_train(y_train==c1) = 1;
                y_train(y_train==c2) = -1;
                ranking_temp = mRMR(predictors(ind_train,:) , y_train, 20);
                ranking_ch = [ranking_ch;ranking_temp(1:20)];

            end
        end

        ranking_ch = unique(ranking_ch(:));

        predictors_this = predictors(:,ranking_ch);

        template = templateSVM('KernelFunction', 'gaussian', ...
            'PolynomialOrder', [], 'KernelScale', 10/80*length(ranking_ch), ...
            'BoxConstraint', 1, 'Standardize', true);

        Mdl = fitcecoc(predictors_this(cvp.trainStore{i}, :), response(cvp.trainStore{i}, :), ...
            'Learners', template, ...
            'Coding', 'onevsone', ...
            'ClassNames', [1; 2; 3; 4; 5]);

        [~, validationScores] = predict(Mdl, predictors_this(cvp.testStore{i}, :));
        hingeloss_test_ch{i} = round(validationScores,4);
        [~, prediction{i}] = max(validationScores, [], 2);

    end
end

save('fe_all_output.mat')

%%

clear hingeloss_traintest true_label fold_number
hingeloss_traintest{3,sub} = [];
for i = 1:sub
    for ch = 1:3
        fold_temp = [];
        for j=1:sub
            hingeloss_traintest{ch,i} = [hingeloss_traintest{ch,i},hingeloss_cvtrain{ch}{i,j}'];
            fold_temp = [fold_temp,j*ones(1,size(hingeloss_cvtrain{ch}{i,j},1))];
        end
        true_label{ch,i} = response(:);
        fold_number{ch,i} = fold_temp(:);
    end
end
CV_number = sub;

save('output_fe.mat', 'true_label','fold_number',"hingeloss_traintest",'CV_number')

