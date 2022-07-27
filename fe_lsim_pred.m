clear
close all

close all
clc
clear
addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets
dataset_dir = [mydir(1:idcs(end-1)-1),'/DataSets/sleepedf20/'];
results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)];

addpath([mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'])
mkdir(results_dir)

load('output_fe.mat')

load('fe_channels.mat')

clear hingeloss_traintest

%% 5-class LSIM


for ch_eeg = 1:3

    predictors = COM{ch_eeg,1}';
    channel_num = size(predictors,1);

    %% training 70-channel LSIM
    close all

    clear lsim_gmm_para_all transitions_matrices_all coupling_tetha_all pi_0_all AIC_all log_likelihood_all BIC_all
    clear fe_traintest

    C = channel_num;


    for ch = 1:C
        for i = 1:CV_number

            this_fold_number = fold_number{1,i};
            counter = 0;
            for j=1:CV_number
                this_set = this_fold_number==j;
                if i==j
                    continue
                end
                counter = counter+1;
                fe_traintest{ch, i,counter} = predictors(ch,this_set) ;
                temp_label = true_label{ch_eeg,i}(this_set)' ;
                channel_states{ch, i,counter} = temp_label(:)';
            end
        end
    end




    extra.plot = 1;
    extra.check_convergence=0;
    extra.sigma_diag = 1;
    sigma_diag = num2str(extra.sigma_diag);
    extra.sup_learn_flag =1;
    extra.auto_gmm = 1;

    num_gmm_component_max = 5;


    for i = 1:CV_number
        clc
        close all
        disp(i)

        max_itration = 50;
        num_gmm_component = ones(1,C)*num_gmm_component_max;
        [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady,coupling_tetha_IM_converge, kl_dist1, acc1] = ...
            lsim_supervised_fast( squeeze(fe_traintest(:, i, :)) , squeeze(channel_states(:, i, :)) , num_gmm_component , max_itration , extra);

        lsim_gmm_para_all{i,1} =  lsim_gmm_para;
        transitions_matrices_all{i,1} = transition_matrices_convex_comb;
        coupling_tetha_all{i,1} = coupling_tetha_convex_comb;
        pi_0_all{i,1} = pi_0_lsim ;
        AIC_all{i,1} = acc1;
        log_likelihood_all{i,1} =log_likelihood;
        BIC_all{i,1} = kl_dist1;
        coupling_tetha_IM_converge_all{i,1} = coupling_tetha_IM_converge;

        ind_selected{i,1} = find(coupling_tetha_convex_comb(:,1)>1/C);
        max_itration = 100;
        num_gmm_component = ones(1,length(ind_selected{i,1}))*num_gmm_component_max;
        [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady,coupling_tetha_IM_converge, kl_dist2, acc2] = ...
            lsim_supervised_fast( squeeze(fe_traintest(ind_selected{i,1}, i, :)) , squeeze(channel_states(ind_selected{i,1}, i, :)) , num_gmm_component , max_itration , extra);

        lsim_gmm_para_allt{i,1} =  lsim_gmm_para;
        transitions_matrices_allt{i,2} = transition_matrices_convex_comb;
        coupling_tetha_allt{i,1} = coupling_tetha_convex_comb;
        pi_0_allt{i,1} = pi_0_lsim ;
        AIC_allt{i,1} = acc2;
        log_likelihood_allt{i,1} =log_likelihood;
        BIC_allt{i,1} = kl_dist2;
        coupling_tetha_IM_converge_allt{i,1} = coupling_tetha_IM_converge;

        lsim_gmm_para_all{i,2} = lsim_gmm_para_allt{i,1};
        transitions_matrices_all{i,2} = transitions_matrices_allt{i,1};
        coupling_tetha_all{i,2} = coupling_tetha_allt{i,1};
        pi_0_all{i,2} = pi_0_allt{i,1} ;
        AIC_all{i,2} = AIC_allt{i,1};
        log_likelihood_all{i,2} = log_likelihood_allt{i,1};
        BIC_all{i,2} = BIC_allt{i,1};
        coupling_tetha_IM_converge_all{i,2} = coupling_tetha_IM_converge_allt{i,1};

        save(['efslsim_',num2str(ch_eeg),'ch.mat'],'ind_selected','lsim_gmm_para_all','transitions_matrices_all','coupling_tetha_all','pi_0_all','AIC_all','log_likelihood_all','BIC_all','coupling_tetha_IM_converge_all')

    end



    %% test lsim
    clear fe_traintest

    for ch = 1:C
        for i = 1:CV_number
            this_fold_number = fold_number{1,i};

            for j=1:CV_number
                this_set = this_fold_number==j;
                fe_traintest{ch, i,j}= predictors(ch,this_set) ;
            end
        end
    end


    for i = 1:CV_number
        for j=1:CV_number

            lsim_gmm_para = lsim_gmm_para_all{i,1} ;
            transition_matrices_convex_comb = transitions_matrices_all{i,1} ;
            coupling_tetha_convex_comb = coupling_tetha_all{i,1};
            pi_0_lsim = pi_0_all{i,1} ;

            [~ ,~ , ~ , alpha_T_all]  = forward_backward_lsim( pi_0_lsim , coupling_tetha_convex_comb  , transition_matrices_convex_comb ,  lsim_gmm_para , squeeze(fe_traintest(:, i,j)) );
            predictors_temp = 0;
            for zee = 1:length(alpha_T_all)
                predictors_temp = predictors_temp + log(alpha_T_all{zee,1}+10^-7);
            end
            hingeloss_traintest1{ch_eeg,i} = [hingeloss_traintest1{ch_eeg,i},round(predictors_temp,3)];
        end
    end

    hingeloss_traintest = hingeloss_traintest1;
    save('output_efelsim.mat', 'true_label','fold_number',"hingeloss_traintest",'CV_number')


    for i = 1:CV_number
        for j=1:CV_number

            lsim_gmm_para = lsim_gmm_para_all{i,2} ;
            transition_matrices_convex_comb = transitions_matrices_all{i,2} ;
            coupling_tetha_convex_comb = coupling_tetha_all{i,2};
            pi_0_lsim = pi_0_all{i,2} ;

            [~ ,~ , ~ , alpha_T_all]  = forward_backward_lsim( pi_0_lsim , coupling_tetha_convex_comb  , transition_matrices_convex_comb ,  lsim_gmm_para , squeeze(fe_traintest(ind_selected{i,1}, i,j)) );
            predictors_temp = 0;
            for zee = 1:length(alpha_T_all)
                predictors_temp = predictors_temp + log(alpha_T_all{zee,1}+10^-7);
            end
            hingeloss_traintest2{ch_eeg,i} = [hingeloss_traintest2{ch_eeg,i},round(predictors_temp,3)];
        end
    end

    hingeloss_traintest = hingeloss_traintest2;
    save('output_efefslsim.mat', 'true_label','fold_number',"hingeloss_traintest",'CV_number')
end


