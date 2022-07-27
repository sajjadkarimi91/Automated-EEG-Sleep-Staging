clear
close all


fname=mfilename('fullpath');
[pathstr,~,~]=fileparts(fname);
addpath([pathstr,'/scatnet-0.2-master']) ;
current_position = cd;

Fs = 100; % sampling rate

sub = 20; % total subjects
epoch = 30;
len = epoch*3; % each feature is extracted from 90s EEG signal;
boundary = 1; 
% 1: avoiding edge effects by considering a much longer signal temporarily 
% 0: the edge artifacts may contaminate the part of the feature we are interested in.

DMdim = 80; % compute the first (DMdim+1) eigenvectors of transition matrix


num_signal = min(2*sub,39)*2;

Channels = cell((num_signal)/2, 2);
label = cell((num_signal)/2, 1);


for caseNo = 1:min(2*sub,39)
    clear input1
    clear input2
    clear STAGE
    
    name=[pathstr,'/SLEEP_EDF_mfile/',num2str(caseNo),'_','PhysioMAT'];

    addpath(name);
    load('STAGE.mat')
    load('PSG1.mat');
    load('PSG2.mat');
    before = (2+boundary); 
    after =  boundary; 
    [x1,x2,num_STAGE,~]=truncated_rawPSG_HYP_SC(STAGE,PSG1,PSG2,(60+before),(60+after)); % include only 30 mins before and after sleep
    current_position=cd;
    cd(name);
    
 
     rmpath(name);
     cd(current_position);
     RR = rem(length(x1),Fs*epoch);
     x1 = x1(1:length(x1)-RR);

     RR = rem(length(x2),Fs*epoch);
     x2 = x2(1:length(x2)-RR);

    label{caseNo} = num_STAGE;

    Channels{caseNo, 1} = x1; % FPZCZ
    Channels{caseNo, 2} = x2; % PZOZ
    
    if (length(x1)/100/30~=length(num_STAGE))
        error('error');
    end
    if (length(x2)/100/30~=length(num_STAGE))
        error('error');
    end 
end
clear x1 x2 truncated_PSG1 truncated_PSG2 PSG1 PSG2 num_STAGE
 

Info.PID = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,...
    11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20]'; % 
Info.PID = Info.PID(1:min(sub*2,39));

%% stacking the signals for the scattering transform

for i = 1:size(Channels, 1)
    
   Channels{i, 1} = buffer_past(Channels{i, 1},Fs,epoch,boundary); % FPZCZ
   Channels{i, 2} = buffer_past(Channels{i, 2},Fs,epoch,boundary); % PZOZ

   
    if (boundary == 1)
    label{i} = label{i}(4:end-1);
    elseif (boundary == 0)
    label{i} = label{i}(3:end);    
    end
    Channels{i, 1} = Channels{i, 1}(:, 1:end);
    Channels{i, 2} = Channels{i, 2}(:, 1:end);
   
    if (length(Channels{i, 1}(:))/100/30/((len+2*boundary*epoch) / 30)~=length(label{i}))
        error('error');
    end
    if (length(Channels{i, 2}(:))/100/30/((len+2*boundary*epoch) / 30)~=length(label{i}))
        error('error');
    end 
    
end

%% scattering transform

% add path
%run('C:\Users\John\OneDrive\Code\ScatteringTransform\scatnet-0.2\addpath_scatnet.m')
name = [pathstr,'\scatnet-0.2-master\addpath_scatnet.m'];
run(name)
% default filter bank with averaging scale of 2048 samples.
T = 2^11;

% First-order filter bank with 8 wavelets per octave.
% Second-order filter bank with 1 wavelet per octave.
filt_opt.Q = [2 2];

% Calculate maximal wavelet scale so that largest wavelet will be of bandwidth T.
filt_opt.J = T_to_J(T, filt_opt);

% Only compute zeroth-, first- and second-order scattering.
scat_opt.M = 2;

% Prepare wavelet transforms to use in scattering.
[Wop, ~] = wavelet_factory_1d(Fs * len, filt_opt, scat_opt);

% Compute the scattering coefficients.
for i = 1:size(Channels, 1)
    for j = 1:size(Channels, 2)
        Channels{i, j} = bsxfun(@minus, Channels{i, j}, median(Channels{i, j}));
        Channels{i, j} = squeeze(format_scat(log_scat(renorm_scat(scat(reshape(Channels{i, j}, [size(Channels{i, j}, 1), 1, size(Channels{i, j}, 2)]), Wop)))));
    end
end


%% organization and indexing

t = cell(size(Channels, 1), 1);
subjectId = cell(size(Info.PID));
for i = 1:size(Channels, 1)
    id = find(label{i} > 0);
    for j = 1:size(Channels, 2)
    
    Channels{i, j} = Channels{i, j}(:,:,id); % removing the features corresponding to MOVEMENT 
    
    end
    t{i} = full(ind2vec(double(label{i}(id)')))';
    subjectId{i} = repelem(Info.PID(i), length(t{i}))';
end

[X,Y] = my_cell2mat(Channels);
if (boundary ==1) % retaining 90s feature
 X = X(:,4:12,:);
 Y = Y(:,4:12,:);
end

X = get_vec_feature(X); 
Y = get_vec_feature(Y);
clear Channels

%% multiview DM or CCA
dim_reduction_method = 2; 
%0: CCA  
%1: DM 
%2: multiview DM 




 
if (dim_reduction_method==0)
    display('running CCA . . .') 
    X = X - repmat(mean(X),size(X,1),1);
    Y = Y - repmat(mean(Y),size(Y,1),1);
    [U, ~, V] = svds((X')*Y, 50);
    XU = X * real(U);
    YV = Y * real(V);
    COM = [XU, YV];
    clear XU YV U V
    display('End of CCA')
elseif (dim_reduction_method==1)
     [Dis1] = squareform(pdist(X,'euclidean'));
     [Dis2] = squareform(pdist(Y,'euclidean'));
     clear X Y
     Dis1 = Dis1.^2;
     Dis2 = Dis2.^2;
     epsilon1 = quantile(Dis1,0.01);
     epsilon2 = quantile(Dis2,0.01);

     display('running DM . . . ')
     [V1, E1, V2, E2] = getDM(Dis1, Dis2,epsilon1,epsilon2,DMdim);
     display('ending DM . . . ')
     clear  Dis1 Dis2 epsilon1 epsilon2
    elseif (dim_reduction_method==2)

     [Dis1] = squareform(pdist(X,'euclidean'));
     [Dis2] = squareform(pdist(Y,'euclidean'));
     clear X Y
     Dis1 = Dis1.^2;
     Dis2 = Dis2.^2;
     epsilon1 = quantile(Dis1,0.01);
     epsilon2 = quantile(Dis2,0.01);
     display('running Multiview DM . . . ')
     [PSI1, PSI2, Sigma1, Sigma2] = getMDM(Dis1, Dis2,epsilon1,epsilon2,DMdim);
     display('ending multiview-DM . . . ')
     clear  Dis1 Dis2 epsilon1 epsilon2
end 



%% 
    if (dim_reduction_method==2) % co-clustering (i.e., multiview DM)
    time_step = 0.3;  
    % part 1
    vecSigma1 = max(diag(real(Sigma1)),0).^time_step;
    effective = setdiff(find(vecSigma1>vecSigma1(2)*10^(-2)),1);
    vecSigma1 = vecSigma1(effective);
    PSI1 = real(PSI1(:,effective));
    % part 2
    vecSigma2 = max(diag((real(Sigma2))),0).^time_step;
    effective = setdiff(find(vecSigma2>vecSigma2(2)*10^(-2)),1);
    vecSigma2 = vecSigma2(effective);
    PSI2 = real(PSI2(:,effective));
    % concatenation
    COM = [PSI1*diag(vecSigma1/vecSigma1(1)) PSI2*diag(vecSigma2/vecSigma2(1)) ];
    clear PSI1 PSI2  Sigma1 Sigma2 vecSigma1 vecSigma2
    end
    
    if (dim_reduction_method==1) % concatenation of DM
    time_step = 0.3;
    % channel 1
    vecE1 = max(diag(real(E1)),0).^time_step;
    effective = setdiff(find(vecE1>vecE1(2)*10^(-2)),1);
    vecE1 = vecE1(effective);
    V1 = real(V1(:,effective));
    COM_channel1 = V1*diag(vecE1/vecE1(1)); 
    % channel 2
    vecE2 = max(diag(real(E2)),0).^time_step;
    effective = setdiff(find(vecE2>vecE2(2)*10^(-2)),1);
    vecE2 = vecE2(effective);
    V2 = real(V2(:,effective));
    COM_channel2 = V2*diag(vecE2/vecE2(1)); 
    % concatenation
    COM = [COM_channel1 COM_channel2];
    clear V1 E1  V2  E2 vecE1  vecE2  COM_channel1 COM_channel2
    end
    
  
    

%% 5-class SVM
predictors = COM;
clear COM
y = cell2mat(t);
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
        
template = templateSVM('KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], 'KernelScale', 10, ...
    'BoxConstraint', 1, 'Standardize', true);

%cm = cell(sub, 1);
prediction = cell(sub, 1);
parfor i = 1:sub
    Mdl = fitcecoc(predictors(cvp.trainStore{i}, :), response(cvp.trainStore{i}, :), ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', [1; 2; 3; 4; 5]);
    
    [~, validationScores] = predict(Mdl, predictors(cvp.testStore{i}, :));
    [~, prediction{i}] = max(validationScores, [], 2);
    %[~, cm{i}, ~, ~] = confusion(y(cvp.testStore{i}, :)', validationScores');
end

each_night_prediction = cell(min(39,2*sub), 1);
for i = 1:sub
    if (i<20)
    P =  prediction{i}; 
    if (length(P)~=size(t{2*i-1},1)+size(t{2*i},1))
        error('error')
    else
       P1 = P(1:size(t{2*i-1},1)); 
       P2 = P(size(t{2*i-1},1)+1:end);
       j1 = 2*i-1;
       j2 = 2*i;
       each_night_prediction{j1} = P1;
       each_night_prediction{j2} = P2; 
    end
    elseif (i==20)
    P =  prediction{i};     
    if (length(P)~=size(t{2*i-1},1))
        error('error')
    else
       P1 = P(1:size(t{2*i-1},1)); 
       each_night_prediction{2*i-1} = P1;
    end 
    else
        error('error');
    end      
end

cm = cell(min(39,2*sub), 1);
for i = 1:min(2*sub,39)
    if (i==34)
     mid1 = 277+60+149+1;
     mid2 = 818-60-149-1;
     YY = t{i};
     VV = each_night_prediction{i};
     VV = full(ind2vec(double(VV'),5))';
     YY= [YY(1:mid1,:); YY(mid2:end,:)];
     VV= [VV(1:mid1,:); VV(mid2:end,:)];
    [~, cm{i}, ~, ~] = confusion(YY', VV');  

    elseif (i==30)
     mid1 = 118+60+149;
     mid2 = 775-60-149;
     YY = t{i};
     VV = each_night_prediction{i};
     VV = full(ind2vec(double(VV'),5))';
     YY= [YY(1:mid1,:); YY(mid2:end,:)];
     VV= [VV(1:mid1,:); VV(mid2:end,:)];
    [~, cm{i}, ~, ~] = confusion(YY', VV');  
    else
     YY = t{i};
     VV = each_night_prediction{i}; 
     VV = full(ind2vec(double(VV'),5))';
    [~, cm{i}, ~, ~] = confusion(YY', VV');    
    end
     
end

ConfMat = zeros(5);
for i = 1:size(cm,1)
    ConfMat = ConfMat + cm{i};
end
acc = sum(diag(ConfMat)) / sum(ConfMat(:));

disp(['Accuracy = ' num2str(acc)])

% ConfMat(i, j) means true class i and output class j

%% metrics

SUM=sum(ConfMat,2);
nonzero_idx=find(SUM~=0);
normalizer=zeros(5,1);
normalizer(nonzero_idx)=SUM(nonzero_idx).^(-1);
matHMM=diag(normalizer)*ConfMat;
normalized_confusion_matrix = matHMM;

SUM=sum(ConfMat,1);
nonzero_idx=find(SUM~=0);
normalizer=zeros(5,1);
normalizer(nonzero_idx)=SUM(nonzero_idx).^(-1);
normalized_sensitivity_matrix=ConfMat*diag(normalizer);

recall = diag(normalized_confusion_matrix);
precision = diag(normalized_sensitivity_matrix);

F1_score = 2*(recall.*precision)./(recall+precision);
Macro_F1 = mean(F1_score);

TOTAL_EPOCH = sum(sum(ConfMat));
ACC = sum(diag(ConfMat))/TOTAL_EPOCH;
EA = sum(sum(ConfMat,1).*sum(transpose(ConfMat),1))/TOTAL_EPOCH^2;
kappa = (ACC-EA)/(1-EA);

output = cell(8, 9);
output(1, 2:end) = {'Predict-W', 'Predict-REM', 'Predict-N1', 'Predict-N2', 'Predict-N3', 'PR', 'RE', 'F1'};
output(2:6, 1) = {'Target-W', 'Target-REM', 'Target-N1', 'Target-N2', 'Target-N3'};
output(2:6, 2:6) = num2cell(ConfMat);
output(2:6, 7) = num2cell(precision);
output(2:6, 8) = num2cell(recall);
output(2:6, 9) = num2cell(F1_score);
output(8, 1:3) = {['Accuracy: ' num2str(ACC)], ['Macro F1: ' num2str(Macro_F1)], ['Kappa: ' num2str(kappa)]};
time = clock;
%xlswrite(['C:\Users\John\OneDrive\Code\Sleep\Metrics' num2str(time(4)) num2str(time(5)) '.xls'], output);

%%
SUM=sum(ConfMat,2);
nonzero_idx=find(SUM~=0);
normalizer=zeros(5,1);
normalizer(nonzero_idx)=SUM(nonzero_idx).^(-1);
matHMM=diag(normalizer)*ConfMat;
normalized_confusion_matrix = matHMM;

SUM=sum(ConfMat,1);
nonzero_idx=find(SUM~=0);
normalizer=zeros(5,1);
normalizer(nonzero_idx)=SUM(nonzero_idx).^(-1);
normalized_sensitivity_matrix=ConfMat*diag(normalizer);

recall = diag(normalized_confusion_matrix);
precision = diag(normalized_sensitivity_matrix);

F1_score = 2*(recall.*precision)./(recall+precision);
Macro_F1 = mean(F1_score);

TOTAL_EPOCH = sum(sum(ConfMat));
ACC = sum(diag(ConfMat))/TOTAL_EPOCH;
EA = sum(sum(ConfMat,1).*sum(transpose(ConfMat),1))/TOTAL_EPOCH^2;
kappa = (ACC-EA)/(1-EA);

output = cell(8, 9);
output(1, 2:end) = {'Predict-W', 'Predict-REM', 'Predict-N1', 'Predict-N2', 'Predict-N3', 'PR', 'RE', 'F1'};
output(2:6, 1) = {'Target-W', 'Target-REM', 'Target-N1', 'Target-N2', 'Target-N3'};
output(2:6, 2:6) = num2cell(ConfMat);
output(2:6, 7) = num2cell(precision);
output(2:6, 8) = num2cell(recall);
output(2:6, 9) = num2cell(F1_score);
output(8, 1:3) = {['Accuracy: ' num2str(ACC)], ['Macro F1: ' num2str(Macro_F1)], ['Kappa: ' num2str(kappa)]};
time = clock;
%xlswrite(['C:\Users\John\OneDrive\Code\Sleep\Metrics' num2str(time(4)) num2str(time(5)) '.xls'], output);
    ratio = sum(ConfMat,2)/sum(sum(ConfMat));
    matHMM=normalized_confusion_matrix;

figure;
imagesc(matHMM);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(matHMM(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:5);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(matHMM(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors


%ratio=[numel(find(testing_GT==1)) numel(find(testing_GT==2)) numel(find(testing_GT==3)) numel(find(testing_GT==4)) numel(find(testing_GT==5))]'/length(testing_GT);

ratio=cellstr(num2str(ratio*100,'%5.0f%%'));

delta=0.2;
specialaxis=[1 1+delta 2 2+delta 3 3+delta 4 4+delta 5 5+delta];
set(gca,'XTick',1:5,...                         %# Change the axes tick marks
        'XTickLabel',{'Awake','REM','N1','N2','N3'},...  %#   and tick labels
        'YTick',specialaxis,...
        'YTickLabel',{'Awake',ratio{1},'REM',ratio{2},'N1',ratio{3},'N2',ratio{4},'N3',ratio{5}},...
        'TickLength',[0 0]);
set(gca,'XAxisLocation','top');
%ylabel('Ground Truth');
xlabel('normalized confusion matrix (SVM)', 'fontweight','bold');   