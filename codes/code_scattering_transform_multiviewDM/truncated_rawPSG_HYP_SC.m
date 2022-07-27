function [truncated_feature1,truncated_feature2, num_truncated_HYP,new_move_point]=truncated_rawPSG_HYP_SC(oversampled_GT,feature1,feature2,before,after)
only_sleep = 1; % Most literatures do not take the truncation into considerstion
[num_oversampled_GT]=translation_ann(oversampled_GT);


% SC25 EEG recording time is longer than the Hypnogram recording time
%if (length(feature1)<length(num_oversampled_GT)*30*100)
%     ratio_miss_slot = ((length(feature1)-length(num_oversampled_GT)*30*100)/30/100);
%end

if (length(feature1)>length(num_oversampled_GT)*30*100)
    feature1 = feature1(1:length(num_oversampled_GT)*30*100);
    feature2 = feature2(1:length(num_oversampled_GT)*30*100);
end


if (only_sleep == 1)
    transition_occur=0;
    first_stage=num_oversampled_GT(1);
    tt = 2;
    aa = 20; %900; %20
    while(transition_occur==0)
        if (num_oversampled_GT(tt)~=first_stage) && (tt>aa)
            transition_occur=1;
            transition_right_time=tt;
        end
        tt=tt+1;
    end
    
    if (transition_right_time-before>aa)
        record_start = transition_right_time-before; % units: 30 second
    else
        display('touch lower bound')
        record_start=aa ;
    end
    
    
    
    %%
    list=1:length(num_oversampled_GT);
    inx_awake=list(num_oversampled_GT==1);
    max_inx_awake=max(inx_awake);
    
    awake=1;
    TT=max_inx_awake;
    while(awake==1)
        if (num_oversampled_GT(TT)~=1)
            awake=0;
            sleep_end_time=TT;
        end
        TT=TT-1;
    end
    
    record_end=sleep_end_time+min(after,length(num_oversampled_GT)-sleep_end_time); % units: 30 second
else
    display('All recording!')
    record_start = 1;
    record_end = length(num_oversampled_GT);
end
%%
num_truncated_HYP=num_oversampled_GT(record_start:record_end);
move_point = find(oversampled_GT=='M');
%uncertain_point = find(oversampled_GT=='?');
%move_point = sort(union(move_point,uncertain_point));
idx1 = find(move_point>record_start-1);
idx2 = find(move_point<record_end+1);
if (numel(idx1)~=numel(move_point))
    error('ERROR3.1')
end
if (numel(idx2)~=numel(move_point))
    error('ERROR3.2')
end
idx12=union(idx1,idx2);
move_point = sort(move_point(idx12));
new_move_point = move_point-record_start+1;

if (length(feature1)>10)
    truncated_feature1=feature1((record_start-1)*30*100+1:record_end*30*100);
    truncated_feature2=feature2((record_start-1)*30*100+1:record_end*30*100);
else
    truncated_feature1= [];
    truncated_feature2= [];
    display('length(feature1)<10 ############')
end

% SSSSSSS =(record_start-1)*30*100+1
% TTTTTTT = record_end*30*100
end