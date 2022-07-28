function [PSI1, PSI2,Sigma1,Sigma2] = getMDM(Dis1, Dis2, epsilon1, epsilon2, DMdim)

% LoadParameters ;

%% affinity matrix (channel 1)


     scaling_vector = epsilon1;
% % %      N =size(Dis1,1);
% % %      parfor n =1:N
% % %      Dis1(n,:) = Dis1(n,:)/scaling_vector(n);
% % %      end
     Dis1 = Dis1./scaling_vector(:);
     W1 = exp(-Dis1);
     clear Dis1 scaling_vector
     W1 = W1-sparse(diag(diag(W1)));
    


%% affinity matrix (channel 2)

     scaling_vector = epsilon2;
%      N =size(Dis2,1);
%      parfor n =1:N
%      Dis2(n,:) = Dis2(n,:)/scaling_vector(n);
%      end
     
     Dis2 = Dis2./scaling_vector(:);
     
     W2 = exp(-Dis2);
     clear Dis2 scaling_vector
     W2 = W2-sparse(diag(diag(W2)));
    


%% running the product of W1 and W2
    W1W2 = W1*W2;
    A12 = sparse(diag(1./sum(W1W2,2)))*W1W2;
   clear W1W2 
   
   W2W1 = W2*W1;
   clear W1 W2
 
   A21 = sparse(diag(1./sum(W2W1,2)))*W2W1;
   clear W2W1
   

%% compute eigenvectors of the big transition matrix
    A12A21 = A12*A21;
    clear A12
    [PSI1,Sigma1] = eigs(A12A21,DMdim+1);
    [~,I] = sort(diag(Sigma1),'descend');
    Sigma1 = Sigma1(I,I);
    PSI1 = PSI1(:,I);
   

   Sigma2 = Sigma1;
   PSI2 = A21*PSI1;
   norm_PSI2=sparse(diag(1./sqrt(sum(abs(PSI2).^2,1))));
   PSI2= PSI2*norm_PSI2;


end
    
 




