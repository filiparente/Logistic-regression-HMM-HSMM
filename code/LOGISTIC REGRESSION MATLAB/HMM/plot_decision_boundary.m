function plot_decision_boundary(betas)
    xx1 = linspace(-15,15);
    if (betas(3,1)-betas(3,2)) ~= 0
        xx2 = ((betas(1,2)-betas(1,1))+(betas(2,2)-betas(2,1))*xx1)/(betas(3,1)-betas(3,2));
    else
        xx2 = ((betas(1,2)-betas(1,1))+(betas(2,2)-betas(2,1))*xx1)/0.00001;  
        %line(ones(2,1)*(betas(1,2)-betas(1,1))+(betas(2,2)-betas(2,1)), get(gca, 'ylim'));
    end
    plot(xx1,xx2);
    ylim([-10,10])
end