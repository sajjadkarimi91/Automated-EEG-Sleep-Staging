
%% load & preprocess EEG data
clc

sel_elec = 1:3;
fix_min = 30;
srate = 100;
fc1 = 0.05;
fc2 = 48;
scale_factor = 200;

Folder_Info = dir([dataset_dir,'*.edf']);

for sub_num = 0:sub-1
    for d_rec=1:2

        if sub_num<10
            file_name = ['SC40',num2str(sub_num),num2str(d_rec),'E0-PSG.edf'];
        else
            file_name = ['SC4',num2str(sub_num),num2str(d_rec),'E0-PSG.edf'];
        end

        for i=1:size(Folder_Info,1)
            if strcmp(Folder_Info(i).name,file_name)
                break
            end
        end
        try

            file_name_hyp = [dataset_dir,Folder_Info(i+1).name];

            hypnogram = processEDFxHypnogram( file_name_hyp );
            ind_slp = find(~(hypnogram=='W'|hypnogram=='?'));
            index_start = max(1,ind_slp(1)-2*fix_min);
            index_stop = min(length(hypnogram),ind_slp(end)+2*fix_min);

            if  contains(hypnogram(:)','M')||contains( hypnogram(:)','?')
                yu=0;
            end

            [hdr, record]  = edfread([dataset_dir,file_name]);

            subj = record(sel_elec,:)';
            EEG_data = subj';
            %eegplot(EEG_data, 'srate', srate);

            EEG_data = EEG_data - mean(EEG_data,2);
            EEG_data = sjk_eeg_filter(EEG_data,srate ,fc1,fc2);

            EEG_clean  = EEG_data;
            %eegplot(EEG_clean, 'srate', srate);
            all_hypnogram{sub_num+1,d_rec} = hypnogram;
            all_hypnogram_L(sub_num+1,d_rec) = length( hypnogram )*30;
            all_record{sub_num+1,d_rec} = EEG_clean';

        end
    end
end

if sub>14
    all_hypnogram_new =  all_hypnogram;
    all_hypnogram_new{14,1} = all_hypnogram{sub,1};
    all_hypnogram_new{14,2} = all_hypnogram{sub,2};
    all_hypnogram_new{sub,1} = all_hypnogram{14,1};
    all_hypnogram_new{sub,2} = all_hypnogram{14,2};
    all_hypnogram = all_hypnogram_new;

    all_record_new =  all_record;
    all_record_new{14,1} = all_record{sub,1};
    all_record_new{14,2} = all_record{sub,2};
    all_record_new{sub,1} = all_record{14,1};
    all_record_new{sub,2} = all_record{14,2};
    all_record = all_record_new;

    clear all_hypnogram_new all_record_new
end
