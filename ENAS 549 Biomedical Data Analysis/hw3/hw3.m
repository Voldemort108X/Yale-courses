%% problem 4
i = 1:1:20;
beta = 2;
sigma = 10;
X = 5*i;
Z = 10*sqrt(X);

eta = beta*X;

nrep = 1000;
for idx = 1:1:nrep
    eta_noisy = eta + sigma*randn(1, size(X,2));

    beta_est_model1(idx) = sum(eta_noisy.*X)/sum(X.^2);
    beta_est_model2(idx) = (sum(Z.*eta_noisy)*sum(X.*Z)-sum(Z.^2)*sum(X.*eta_noisy))/((sum(X.*Z))^2-sum(X.^2)*sum(Z.^2));
    gamma_est_model2(idx) = (sum(X.*eta_noisy)*sum(X.*Z)-sum(X.^2)*sum(Z.*eta_noisy))/((sum(X.*Z))^2-sum(X.^2)*sum(Z.^2));
end

beta_est_model1_mean = mean(beta_est_model1);
beta_est_model1_std = std(beta_est_model1);

beta_est_model2_mean = mean(beta_est_model2);
beta_est_model2_std = std(beta_est_model2);

gamma_est_model2_mean = mean(gamma_est_model2);
gamma_est_model2_std = std(gamma_est_model2);
beta_est_model1_mean = mean(beta_est_model1);

%% problem 5
n = 16;
beta0 = 10;
beta1 = 1;
sigma = 2;
X = 1:1:n;
% X = [1,1,1,1,1,1,1,1,16,16,16,16,16,16,16,16];
nrep = 1000;

eta = beta0 + beta1*X;
Z = ones(1, size(X,2));
for idx = 1:1:nrep
    eta_noisy = eta + sigma*randn(1, size(X,2));
    beta0_est(idx) = (sum(X.*eta_noisy)*sum(X.*Z)-sum(X.^2)*sum(Z.*eta_noisy))/((sum(X.*Z))^2-sum(X.^2)*sum(Z.^2));
    beta1_est(idx) = (sum(Z.*eta_noisy)*sum(X.*Z)-sum(Z.^2)*sum(X.*eta_noisy))/((sum(X.*Z))^2-sum(X.^2)*sum(Z.^2));
end

beta0_est_mean = mean(beta0_est);
beta0_est_var = var(beta0_est);
beta1_est_mean = mean(beta1_est);
beta1_est_var = var(beta1_est);


%% problem 6
n = 10;
X = 1:1:n;
beta = 2;
mu_beta = 2;
sigma_list = [1, 4, 16];
Var_beta_list = [0.1, 1, 10];

n_rep = 1000;

sigma_idx = 1;
for sigma = sigma_list
    var_idx = 1;
    for Var_beta = Var_beta_list
        for i = 1:1:n_rep
            Y = beta * X + sigma*randn(1, size(X,2));
            beta_est(sigma_idx, var_idx, i) = (Var_beta*sum(X.*Y)+mu_beta*sigma^2)/(Var_beta*sum(X.*X)+sigma^2);
        end
    var_idx = var_idx + 1;
    end
    sigma_idx = sigma_idx + 1;
end

for i = 1:1:3
    for j = 1:1:3
        mean_mat(i,j) = mean(beta_est(i,j,:));
        std_mat(i,j) = std(beta_est(i,j,:));
    end
end
