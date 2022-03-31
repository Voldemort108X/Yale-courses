%% problem 4

nrep = 100;
i = 1:1:20;
x = 3*i;

for idx = 1:1:nrep
    
    beta = [100,100,100];
    X = [ones(1, size(i,2)); exp(-0.1*x); exp(-0.01*x)];
    eta = beta * X;
    Y = normrnd(eta, 0.1*eta);
    % Y = poissrnd(eta);

    W = diag(1./(0.1*eta));
    
    beta_wls_3p = inv(X*W*X.')*X*W*Y.';
    beta1_wls_3p(idx) = beta_wls_3p(1);
    beta2_wls_3p(idx) = beta_wls_3p(2);
    beta3_wls_3p(idx) = beta_wls_3p(3);
    
    X_2p = [ones(1, size(i,2)); exp(-0.1*x)];
    beta_wls_2p = inv(X_2p*W*X_2p.')*X_2p*W*Y.';
    beta1_wls_2p(idx) = beta_wls_2p(1);
    beta2_wls_2p(idx) = beta_wls_2p(2);
end

beta1_wls_3p_mean = mean(beta1_wls_3p);
beta1_wls_3p_std = std(beta1_wls_3p);
beta2_wls_3p_mean = mean(beta2_wls_3p);
beta2_wls_3p_std = std(beta2_wls_3p);
beta3_wls_3p_mean = mean(beta3_wls_3p);
beta3_wls_3p_std = std(beta3_wls_3p);

beta1_wls_2p_mean = mean(beta1_wls_2p);
beta1_wls_2p_std = std(beta1_wls_2p);
beta2_wls_2p_mean = mean(beta2_wls_2p);
beta2_wls_2p_std = std(beta2_wls_2p);

plot(i, Y);
hold on
plot(i, beta_wls_3p.' * X);
hold on
plot(i, beta_wls_2p.' * X_2p);
legend('sample data', '3 param model', '2 param model');
xlabel('i');
ylabel('Y');
