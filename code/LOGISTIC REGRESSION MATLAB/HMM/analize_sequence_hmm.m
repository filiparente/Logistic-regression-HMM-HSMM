function [estimated_pi, estimated_T] = analize_sequence_hmm(x, N, dim, n_states, mode)
initial_counts = zeros(n_states,1);
aux_counts = zeros(n_states,1);
transitions = zeros(n_states);
estimated_pi = zeros(n_states,1);
estimated_T = zeros(n_states);


for i=1:N
    prev_state = 0;
    initial_counts(x(1,i)) = initial_counts(x(1,i))+1;

    for j=1:dim
        state = x(j,i);

        if(prev_state ~= 0)      
            transitions(prev_state,state) =  transitions(prev_state,state)+1;
        end
        prev_state = state;
    end
end



%normalize
if strcmp(mode,'normalize')
    estimated_pi = (initial_counts+1e-100)/size(x,2);

    for i=1:n_states
        if sum(transitions(i,:))==0 %if state i does not appear in my decoded sequence at this iteration, then no information about it is retrieved, i.e, transitions and durations become uniform distributions
            estimated_T(i,:) = (1/(n_states-1));
        else
            estimated_T(i,:) = (transitions(i,:))/(sum(transitions(i,:)));
        end

    end
else
    estimated_pi = initial_counts;
    estimated_T = transitions;
end

end

        