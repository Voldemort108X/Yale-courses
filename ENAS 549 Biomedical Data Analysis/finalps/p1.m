%% Problem 1
clear all; close all;

beta1 = 100;
beta2 = 0.01;
beta3 = 200;

x = 20:20:900;
sigma = 20;
n_rep = 200;

b0 = [beta1, beta2, beta3];
y = beta1*sin(beta2*(x+beta3));
f_ols = @(b, x) b(1).*sin(b(2).*(x+b(3)));
for i=1:1:n_rep
    eta = y + sigma * randn(1, size(y,2));
    options=optimset('display','off','Algorithm','levenberg-marquardt'); 
    [b,ss,residual,exitflag,output]=lsqcurvefit(f_ols,b0,x,eta,[],[],options); 
    b_ols(i,:) = b';
    residual_ols(i,:) = residual';
end

% part a
beta1_mean = mean(b_ols(:,1));
beta1_std = std(b_ols(:,1));
beta2_mean = mean(b_ols(:,2));
beta2_std = std(b_ols(:,2));
beta3_mean = mean(b_ols(:,3));
beta3_std = std(b_ols(:,3));

% part b
X = [sin(beta2*x+beta2*beta3);beta1*cos(beta2*x+beta2*beta3).*(x+beta3);beta1*cos(beta2*x+beta2*beta3).*(beta2)]';
cov_ols_pred = inv(X'*X)*sum((y-eta).^2/(45-3));
beta1_std_pred = sqrt(cov_ols_pred(1,1));
beta2_std_pred = sqrt(cov_ols_pred(2,2));
beta3_std_pred = sqrt(cov_ols_pred(3,3));

% part c
corr_b2b3_mat = corrcoef(b_ols(:,2), b_ols(:,3));
corr_b2b3 = corr_b2b3_mat(1,2);

% part d
cov_diag = sqrt(diag(cov_ols_pred));
corr_ols_pred = (1./cov_diag)*(1./cov_diag)'.*cov_ols_pred;

% part e
subplot(2,2,1);
plot(x,eta);
hold on
plot(x, f_ols(b, x));
legend('data','fit');
title('Example data and fit plot');

subplot(2,2,2)
plot(x, residual);
title('Example residual plot');

subplot(2,2,3);
res_ols_mean = mean(residual_ols);
res_ols_std = std(residual_ols);
errorbar(x, res_ols_mean, res_ols_std);
title('Residual v.s. x as mean with error bars')

subplot(2,2,4);
scatter(b_ols(:,2), b_ols(:,3));
title('Correlation of b2 and b3');

saveas(gcf,'p1','png')