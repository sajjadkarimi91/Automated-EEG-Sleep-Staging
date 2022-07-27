function [hypnogram] = processEDFxHypnogram( hyp_file )
%processEDFxHypnogram Reads the annotation file to produce a hypnogram
%   [hypnogram] = processEDFxHypnogram(hyp_file) uses the annotation file
%   downloaded to produce a per-epoch hypnogram with the following labels:
%   W, 1, 2, 3, 4, R, M



% Regular expression search string
rexp = '([SM][\w\?]+\s\w+:\s\d+)';
%st_time_rexp = '(\[\d+:\d+:\d+.\d+)?';

% Define epoch size
epoch_size = 30;

% Read the annotation file and search for strings that have time and sleep
% stage information
hyp_read = fileread(hyp_file);
%
h1 = strfind(hyp_read, 'Sleep stage ');
h2 = strfind(hyp_read, 'Movement time');
h1 = sort([h1,h2]);

LL = length('Sleep stage ');
% Initialize containers for hypnogram value and duration
hyp_v = char.empty(length(h1)-1,0); %value
hyp_d = zeros(size(h1)-1); %duration

% Extract sleep stage and the duration for which it lasts
for h=1:length(h1)-1
    if h==1
        hyp_string = hyp_read(h1(h)-25+LL:h1(h)+LL);
    else
        hyp_string = hyp_read(h1(h-1)+LL:h1(h)+LL);
    end
    
    %     C = textscan(hyp_string, '%s', 'delimiter', '\u0014');
    %     C=C{1};
    %
    % fix for movement time
    if hyp_string(end) =='e' %contains(hyp_string,'Movement time')
        hyp_v(h)='M';
    else
        hyp_v(h) = hyp_string(end);
    end
    
    plus_index = strfind(hyp_string,'+');
    plus_index = plus_index(end);
    hyp_string = hyp_string(plus_index(end)+1:end-LL-1);
    hyp_string(end)=[];
    for d = 1:6
        if isempty(str2num(hyp_string(end-d-1:end)))
            hyp_d(h) = str2num(hyp_string(end-d:end));
            break
        end
    end
    % save hyp durations
    %     hyp_d(h) = str2num(C{3});
    
    
end

% Total (round) number of epochs
number_of_epochs = sum(hyp_d)/epoch_size;

% Container for hypnogram
hypnogram = char.empty(number_of_epochs,0);

% Using the duration of each stage, assign hypnogram value for each 30s
% epoch in the vector
idx=0;
for h=1:length(hyp_d)
    ep_end = hyp_d(h)/epoch_size;
    hypnogram(idx+1:idx+ep_end,1)=hyp_v(h);
    idx=idx+ep_end;
end

%{
% Convert to AASM if that is the classification_mode
% Conversion is as follows
% M  -> W
% 4  -> 3
% Rest are same: W,1,2,3,R
if strcmp(classification_mode,'AASM')
    hypnogram(hypnogram=='M')='W';
    hypnogram(hypnogram=='4')='3';
end
%}