function [GT] = translation_ann(annotation)


start_question = min(find(annotation=='?'));
annotation(start_question:end)=[];
L=length(annotation);
tempGT=zeros(L,1);


%unique(annotation)

idx_M = find(annotation=='M');
idx_A = find(annotation=='W');
idx_R = find(annotation=='R');
idx_N1 = find(annotation=='1');
idx_N2 = find(annotation=='2');
idx_N3 = find(annotation=='3');
idx_N4 = find(annotation=='4');

if (numel(find(annotation=='?'))>0)
    error('error in the step of translation')
end

%RATIO = [ numel(idx_A) numel(idx_R) numel(idx_N1) numel(idx_N2) numel(idx_N3)+numel(idx_N4)]/L;

tempGT(find(annotation=='M'))=0;
tempGT(find(annotation=='W'))=1;
tempGT(find(annotation=='R'))=2;
tempGT(find(annotation=='1'))=3;
tempGT(find(annotation=='2'))=4;
tempGT(find(annotation=='3'))=5;
tempGT(find(annotation=='4'))=5;






GT=tempGT;
% if (numel(find(tempGT==100))>0);
%  move_point = find(tempGT==100)';
%   for kk= 1:numel(move_point)
%       KK=move_point(kk);
%       Local = tempGT(KK-1:KK+1);
%       GT(KK) = min(Local);
%   end 
% end



end


% 
% function [ GT ] = transation_ann( annotation )
% 
% L=length(annotation);
% GT=zeros(L,1);
% 
% GT(find(annotation=='M'))=1;
% GT(find(annotation=='W'))=1;
% GT(find(annotation=='R'))=2;
% GT(find(annotation=='1'))=3;
% GT(find(annotation=='2'))=4;
% GT(find(annotation=='3'))=5;
% GT(find(annotation=='4'))=5;
% 
% GT(find(annotation=='?'))=1;
% 
% end