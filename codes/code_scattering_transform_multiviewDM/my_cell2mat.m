function [X,Y] = my_cell2mat(cell_feature,flag_2D)

if nargin ==1
    flag_2D = 0;
end

if flag_2D == 0
    Nx = zeros(size(cell_feature,1),1);
    Ny = zeros(size(cell_feature,1),1);
    Nz = zeros(size(cell_feature,1),1);
    for k = 1:size(cell_feature,1)
        X = cell_feature{k,1};
        [Nx(k),Ny(k),Nz(k)] = size(X);
    end
    
    total_Nz = sum(Nz);
    if (numel(unique(Nx))~=1) || (numel(unique(Ny))~=1)
        error('error');
    end
    Nx = Nx(1);
    Ny = Ny(1);
    
    cum_Nz = cumsum(Nz);
    
    X = zeros(Nx,Ny,total_Nz);
    X(:,:,1:Nz(1)) = cell_feature{1,1};
    for k = 2:size(cell_feature,1)
        temp = cell_feature{k,1};
        X(:,:,cum_Nz(k-1)+1:cum_Nz(k)) =  temp;
    end
    
    Y = zeros(Nx,Ny,total_Nz);
    Y(:,:,1:Nz(1)) = cell_feature{1,2};
    for k = 2:size(cell_feature,1)
        temp = cell_feature{k,2};
        Y(:,:,cum_Nz(k-1)+1:cum_Nz(k)) =  temp;
    end
    
else
    
    Nx = zeros(size(cell_feature,1),1);
    Nz = zeros(size(cell_feature,1),1);
    for k = 1:size(cell_feature,1)
        X = cell_feature{k,1};
        [Nx(k),Nz(k)] = size(X);
    end
    
    total_Nz = sum(Nz);
    if (numel(unique(Nx))~=1)
        error('error');
    end
    Nx = Nx(1);
    cum_Nz = cumsum(Nz);
    
    X = zeros(Nx,total_Nz);
    X(:,1:Nz(1)) = cell_feature{1,1};
    for k = 2:size(cell_feature,1)
        temp = cell_feature{k,1};
        X(:,cum_Nz(k-1)+1:cum_Nz(k)) =  temp;
    end
    
    Y = zeros(Nx,total_Nz);
    Y(:,1:Nz(1)) = cell_feature{1,2};
    for k = 2:size(cell_feature,1)
        temp = cell_feature{k,2};
        Y(:,cum_Nz(k-1)+1:cum_Nz(k)) =  temp;
    end
    
end

end



