function transition_matrix = generate_random_transition_matrix(N)
    transition_matrix = zeros(N);
    
    for i=1:N
        row = randsample(100, N);
        row = row/sum(row);

        transition_matrix(i, :) = row;
    end
 
end

