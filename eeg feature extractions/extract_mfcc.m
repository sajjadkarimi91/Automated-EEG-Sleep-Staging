function [mfcc_time_series , index_nan, fbe_time_series] = extract_mfcc(EEG , num_CC , num_filter_bank ,fs , time_window , LF , HF , type_sig)


% Define variables
Tw = time_window*1000 ;                % analysis frame duration (ms)
Ts =Tw;% floor( Tw /4);               % analysis frame shift (ms)
alpha = 0.95;           % preemphasis coefficient
M = num_filter_bank;                 % number of filterbank channels
C = num_CC;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
% LF = 0.5;               % lower frequency limit (Hz)
% HF = 45;              % upper frequency limit (Hz)


mfcc_time_series = [];
fbe_time_series = [];
index_nan =[];

for n = 1:size(EEG,1)
    
    if(strcmp(type_sig,'eeg'))
    [ MFCCs, FBEs, frames ]= ...
        mfcc_eeg( EEG(n,:)', fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L , 10);
    else
        [ MFCCs, FBEs, frames ]= ...
        mfcc_ecg( EEG(n,:)', fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L , 10);
  
    end
    mfcc_time_series = cat(1 , mfcc_time_series ,MFCCs);
    fbe_time_series = cat(1 , fbe_time_series ,FBEs);
    index_nan = cat(1, index_nan ,find(isnan(sum(frames,1))) );
    
end


