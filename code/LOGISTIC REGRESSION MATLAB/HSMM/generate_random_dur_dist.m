function dur_probs = generate_random_dur_dist(N,D)
    dur_probs = zeros(N, D);

    for i=1:N
        row = randsample(1000, D);

        dur_probs(i, :) = row/sum(row);
    end
end