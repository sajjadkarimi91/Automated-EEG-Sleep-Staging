
function [eeg_features, vec_eeg_features]= eeg_features_channel(x , windowSize , srate)

%% multiscale permutation entropy

order = 4;
delay = 1;

windowSize_n = windowSize;
[~,outdata_2] = PE( x(:,1)', delay, order, windowSize_n-delay*order );
mpe_features(1,:) = outdata_2(1,windowSize_n:windowSize_n:end);

windowSize_n = windowSize/2;
xmv = movmean(x(:,1)',[1,0]);
[~,outdata_2] = PE( xmv(2:2:end) , delay, order, windowSize_n-delay*order );
mpe_features(2,:) = outdata_2(1,windowSize_n:windowSize_n:end);

windowSize_n = windowSize/3;
xmv = movmean(x(:,1)',[2,0]);
[~,outdata_2] = PE( xmv(3:3:end), delay, order, windowSize_n-delay*order );
mpe_features(3,:) = outdata_2(1,windowSize_n:windowSize_n:end);



%% Statistical features

X_wdw = reshape(x(:,1),windowSize,[]);
stat_features(1,:) = std(X_wdw);
stat_features(2,:) = skewness(X_wdw);
stat_features(3,:) = kurtosis(X_wdw);


%% AR features

X_wdw = reshape(x(:,1),windowSize,[]);

for t = 1:size(X_wdw,2)
    y = X_wdw(:,t)';
    [mb,ref1] = ar(y(:),8,'burg');
    ar_features_1(:,t) = ref1(1,2:end);
    ar_features_11(:,t) = mb.A(2:end);
end


ref_features = ar_features_1;
ar_features = ar_features_11;

%% spectrul entropy

X_wdw = reshape(x(:,1),windowSize,[]);
pxx_1 = pwelch(X_wdw,srate,[],srate);
pxx_1 = sqrt(pxx_1(2:31,:));
pxx_1 = pxx_1./sum(pxx_1,1);
pxx_15 = pxx_1(5:end,:)./sum(pxx_1(5:end,:),1);
pxx_19 = pxx_1(9:end,:)./sum(pxx_1(9:end,:),1);


spectrul_entropy_1 = [-sum(pxx_1.*log(pxx_1),1);-sum(pxx_15.*log(pxx_15),1);-sum(pxx_19.*log(pxx_19),1)];
spectrul_entropy = [spectrul_entropy_1];

%% fractional spectral radius (FSR) & Hjorth parameters mobility and complexity

X_wdw = reshape(x(:,1),windowSize,[]);
parfor t = 1:size(X_wdw,2)
    y = X_wdw(:,t)';
    fsrj = fsr_eeg(y(1:2:end),15);
    fsrj_1(:,t) = fsrj;

    [mobility,complexity] = hjorth_parameters(y(1:2:end)');
    Hjorth_1(:,t) = [mobility;complexity];
end


fsrj_features = fsrj_1(1:end-1,:);
Hjorth_features = Hjorth_1;

%% approximate entropy (ApEn)


X_wdw = reshape(x(:,1),windowSize,[]);
c= 0;
for dim = 2:10
    c= c+1;

    parfor t = 1:size(X_wdw,2)
        y = X_wdw(:,t)';
        tau = 3;
        %apen = approximateEntropy(y(:),'Dimension', dim, 'Lag', tau, 'Radius',r );
        r = dim/30*std(y);
        apen = ApEn( dim, r, y, tau );
        apen_1(c,t) = apen;
    end

end



apen_features =apen_1;

%% Lyapunov exponent


X_wdw = reshape(x(:,1),windowSize,[]);
c= 0;
for dim = 4:2:8

    c= c+1;

    parfor t = 1:size(X_wdw,2)
        y = X_wdw(:,t)';
        tau = 2;
        lyapExp  = lyapunovExponent(y(:),srate,'Dimension', dim, 'Lag', tau, 'ExpansionRange',[1,20] );
        lyapExp_1(c,t) = lyapExp ;
    end

end


lyapExp_features = lyapExp_1;


%% correlation dimension

X_wdw = reshape(x(:,1),windowSize,[]);
c= 0;
for dim = 4:2:8

    c= c+1;
    parfor t = 1:size(X_wdw,2)
        y = X_wdw(:,t)';
        tau = 2;
        Np = 20;
        MinR = 1/10*std(y);
        MaxR = 5/10*std(y);

        try
            corDim  = correlationDimension(y(:),'Dimension', dim, 'Lag', tau,'NumPoints',Np ,'MinRadius',MinR,'MaxRadius',MaxR,'NumPoints',Np);
        catch
            MaxR = 10/10*std(y);
            corDim  = correlationDimension(y(:),'Dimension', dim, 'Lag', tau,'NumPoints',Np ,'MinRadius',MinR,'MaxRadius',MaxR,'NumPoints',Np);
        end

        corDim_1(c,t) = corDim ;
    end

end


corDim_features = corDim_1;

%%

num_CC =8;
num_filter_bank = 20;
type_sig = 'eeg';
time_window = 30;
LF = 1;
HF = 30;
srate = 100;

% power & relative power features

[~,wavelet_time_series_n] = extract_wavelet_energys(x' , srate , time_window , time_window );

[mfcc_time_series_eeg , indexnan] = extract_mfcc(x' , num_CC , num_filter_bank ,srate , time_window , LF , HF, type_sig);


mfcc_time_series_eeg = mfcc_time_series_eeg - mean(mfcc_time_series_eeg,2);


%%

eeg_features(1).f = mpe_features;
eeg_features(2).f = stat_features;
eeg_features(3).f = ref_features;
eeg_features(4).f = ar_features;
eeg_features(5).f = spectrul_entropy;
eeg_features(6).f = fsrj_features;
eeg_features(7).f = Hjorth_features;
eeg_features(8).f = apen_features;
eeg_features(9).f = lyapExp_features;
eeg_features(10).f = corDim_features;
eeg_features(11).f = wavelet_time_series_n;
eeg_features(12).f = mfcc_time_series_eeg;


vec_eeg_features = [mpe_features;stat_features;ref_features;ar_features;spectrul_entropy;fsrj_features;Hjorth_features;apen_features;lyapExp_features;corDim_features;wavelet_time_series_n;mfcc_time_series_eeg];

