%% problem 3
i = 1:1:50;
beta1 = 200;
beta2 = 50;
alph1 = 0.1;
alph2 = 0.02;
t = 2*i;
nrep = 1000;

for i = 1:1:nrep

    eta = beta1*exp(-alph1*t) + beta2*exp(-alph2*t);
    Y = poissrnd(eta).';
    X = [exp(-alph1*t); exp(-alph2*t)].';

    % ols approximation
    b_ols = inv(X.'*X)*X.'*Y;
    b1_ols(i) = b_ols(1);
    b2_ols(i) = b_ols(2);

    % wls approximation
    W = diag(1./eta.');
    b_wls = inv(X.'*W*X)*X.'*W*Y;
    
    b1_wls(i) = b_wls(1);
    b2_wls(i) = b_wls(2);
end
%%
psi = diag(Y);

b1_ols_mean = mean(b1_ols);
b2_ols_mean = mean(b2_ols);
b1_ols_std = std(b1_ols);
b2_ols_std = std(b2_ols);
cc_ols = corrcoef(b1_ols, b2_ols);
cc_ols_sample = cc_ols(1,2);

% cov_ols_pred = inv(X.'*X)*(sum((Y-eta.').^2)/(50-2));
cov_ols_pred = inv(X.'*X)*X.'*psi*X*inv(X.'*X);
b1_ols_std_pred = sqrt(cov_ols_pred(1,1));
b2_ols_std_pred = sqrt(cov_ols_pred(2,2));
cc_ols_pred = cov_ols_pred(1,2)/(b1_ols_std_pred * b2_ols_std_pred);

b1_wls_mean = mean(b1_wls);
b2_wls_mean = mean(b2_wls);
b1_wls_std = std(b1_wls);
b2_wls_std = std(b2_wls);
% cc_wls = cov(b1_wls, b2_wls)./(b1_wls_std * b2_wls_std);
cc_wls = corrcoef(b1_wls, b2_wls);
cc_wls_sample = cc_wls(1,2);

% cov_wls_pred = inv(X.'*W*X)*(sum((Y-eta.').^2)/(50-2));
cov_wls_pred = inv(X.'*W*X)*X.'*W*psi*W*X*inv(X.'*W*X);
b1_wls_std_pred = sqrt(cov_wls_pred(1,1));
b2_wls_std_pred = sqrt(cov_wls_pred(2,2));
cc_wls_pred = cov_wls_pred(1,2)/(b1_wls_std_pred * b2_wls_std_pred);
