function A_ = mean_matrices(A)
    %A is a cell with matrices
    n_matrices = length(A);
    dim =  size(A{1});
    n_states = dim(1);
    
    A = cell2mat(A');
    
    A_ = zeros(dim(1),dim(2));

    for i=1:n_states
        A_(i,:) = mean(A(i:n_states:n_states*n_matrices,:)); 
    end
end
