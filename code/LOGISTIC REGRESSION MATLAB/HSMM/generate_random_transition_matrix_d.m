function transition_matrix = generate_random_transition_matrix_d(N)
    transition_matrix = zeros(N);

    for i=1:N     
        transition_matrix(i, :) =  randsample(100, N);
    end

    for i=1:N
        transition_matrix(i,i) = 0.0;
        transition_matrix(i,:) = transition_matrix(i,:)/sum(transition_matrix(i,:));
    end
end