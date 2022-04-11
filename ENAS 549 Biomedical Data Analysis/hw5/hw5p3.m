%% problem 3
beta1 = 100;
beta2 = 0.3;
t = 0:0.2:10;
nrep = 100;
eta = beta1*exp(-beta2*t);

delta = 0.0001;
f_ols = @(b,t) b(1) .* exp(-b(2)*t);
f_wls = @(b,t,w) sqrt(w) * b(1) .* exp(-b(2)*t);

for idx = 1:1:nrep
    y = poissrnd(eta);

    % OLS fitting
    X = [ones(1,size(t,2)); t]';
    Y = log(y+delta)';
    b0_ols = inv(X.'*X)*X.'*Y;
    b0 = [exp(b0_ols(1)); -b0_ols(2)];
    b_ols = lsqcurvefit(f_ols, b0, t, y);
    beta1_ols(idx) = b_ols(1);
    beta2_ols(idx) = b_ols(2);
    
    % WLS fitting
    w = 1./(y+delta);
    z = sqrt(w) .* y;
    b_wls = lsqcurvefit(f_wls, b0, t, z, [], [], [], w);
    beta1_wls(idx) = b_wls(1);
    beta2_wls(idx) = b_wls(2);
    
end

mean_beta1_ols = mean(beta1_ols);
std_beta1_ols = std(beta1_ols);
mean_beta2_ols = mean(beta2_ols);
std_beta2_ols = std(beta2_ols);
mean_beta1_wls = mean(beta1_wls);
std_beta1_wls = std(beta1_wls);
mean_beta2_wls = mean(beta2_wls);
std_beta2_wls = std(beta2_wls);