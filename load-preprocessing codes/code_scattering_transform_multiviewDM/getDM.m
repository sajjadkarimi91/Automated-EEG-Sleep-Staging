function [V1, E1, V2, E2] = getDM(Dis1, Dis2,epsilon1,epsilon2,DMdim)



%% get diffusion on the first sensor   

% get affinity matric
    scaling_vector = epsilon1;
% % %     N = size(Dis1,1);
% % %     parfor n =1:N
% % %     Dis1(n,:) = Dis1(n,:)/scaling_vector(n);
% % %     end
    
    Dis1 = Dis1./scaling_vector(:);
    
    W1 = exp(-Dis1);
    clear Dis1 scaling_vector
    W1 = W1-sparse(diag(diag(W1)));
    
    sum_row = sum(W1, 2);
    sum_row((sum_row==0)) = 1;
    D1 = sparse(diag(1./sum_row)) ;
	A1 = D1 * W1 ;
    clear W1 D1 sum_col sum_row

    
%% get diffusion on the second sensor   

     scaling_vector = epsilon2;
%      N =size(Dis2,1);
%      parfor n =1:N
%      Dis2(n,:) = Dis2(n,:)/scaling_vector(n);
%      end
     Dis2 = Dis2./scaling_vector(:);
     W2 = exp(-Dis2);
     clear Dis2 scaling_vector

     W2 = W2-sparse(diag(diag(W2)));
     sum_row = sum(W2, 2);
     sum_row((sum_row==0)) = 1;
	 D2 = sparse(diag(1./sum_row)) ;
	 A2 = D2 * W2 ;
     clear W2 D2 sum_col sum_row



    % channel 1
    [V1,E1] = eigs(A1,DMdim+1);
    [~, I] = sort(real(diag(E1)), 'descend');
	E1 = (E1(I,I));
	V1 = V1(:,I);
    
    
    % channel 2
 
    [V2,E2] = eigs(A2,DMdim+1);
    [~, I] = sort(real(diag(E2)), 'descend');
	E2 = (E2(I,I));
	V2 = V2(:,I);
       
end
    


    
 




