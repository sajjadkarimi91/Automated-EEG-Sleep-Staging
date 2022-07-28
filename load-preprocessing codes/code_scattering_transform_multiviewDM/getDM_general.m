function [V, E] = getDM_general(Dis_cell,epsilon,DMdim)



%% get diffusion on the first sensor

for ch = 1:length(Dis_cell)

    % get affinity matric
    scaling_vector = epsilon(ch,:);
    Dis_cell{ch,1} = Dis_cell{ch,1}./scaling_vector(:);

    W1 = exp(-Dis_cell{ch,1});
    Dis_cell{ch,1}=[];
    clear scaling_vector
    W1 = W1-sparse(diag(diag(W1)));

    sum_row = sum(W1, 2);
    sum_row((sum_row==0)) = 1;
    D1 = sparse(diag(1./sum_row)) ;
    A1 = D1 * W1 ;
    clear W1 D1 sum_col sum_row

    % channel 1
    [V1,E1] = eigs(A1,DMdim(ch)+1);
    [~, I] = sort(real(diag(E1)), 'descend');
    E1 = (E1(I,I));
    V1 = V1(:,I);

    V{ch,1} = V1;
    E{ch,1} = E1;
end









