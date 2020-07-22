function pi = generate_random_initial_dist(N)
    row = randsample(100, N);
        
    pi = row/sum(row);
end