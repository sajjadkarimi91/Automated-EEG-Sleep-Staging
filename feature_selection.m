clear
clc

load('fslsim_1ch.mat', 'ind_selected')
first_ind = cell2mat( ind_selected(:,2));

load('fslsim_2ch.mat', 'ind_selected')
second_ind =  cell2mat(ind_selected(:,2));

load('fslsim_3ch.mat', 'ind_selected')
third_ind =  cell2mat(ind_selected(:,2));

all_ind = [first_ind;second_ind;third_ind];


all_ind = rem(all_ind,69);
all_ind(all_ind==0) = 69;

[a,b]= hist(all_ind,0.5:1:70);

ind_remove = find(a<40)';

vec_eeg_features_names = zeros(70,1);

vec_eeg_features_names(1:3) = 1;
vec_eeg_features_names(4:6) = 2;
vec_eeg_features_names(7:14) = 3;
vec_eeg_features_names(15:22) = 4;
vec_eeg_features_names(23:25) = 5;
vec_eeg_features_names(26:39) = 6;
vec_eeg_features_names(40:41) = 7;
vec_eeg_features_names(42:50) = 8;
vec_eeg_features_names(51:53) = 9;
vec_eeg_features_names(54:56) = 10;
vec_eeg_features_names(57:60) = 11;
vec_eeg_features_names(61:69) = 12;

[ind_remove,vec_eeg_features_names(ind_remove)]
% eeg_features(1).f = mpe_features;
% eeg_features(2).f = stat_features;
% ****  eeg_features(3).f = ref_features;
% eeg_features(4).f = ar_features;
% eeg_features(5).f = spectrul_entropy;
% ****  eeg_features(6).f = fsrj_features;
% eeg_features(7).f = Hjorth_features;
% ****  eeg_features(8).f = apen_features;
% eeg_features(9).f = lyapExp_features;
% eeg_features(10).f = corDim_features;
% eeg_features(11).f = wavelet_time_series_n;
% eeg_features(12).f = mfcc_time_series_eeg;

