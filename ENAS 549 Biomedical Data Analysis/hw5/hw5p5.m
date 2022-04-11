%% problem 5
bmax = 25;
kd = 10;
sigma = 2;
nrep = 500;
center_list = [1.5,3,6,10,15,30,60,100,150,300];
n_center = size(center_list, 2);

for ctr_idx = 1:1:n_center
    free = center_list(ctr_idx) * logspace(-1,1,21);
    eta = recbinding([bmax; kd], free);
    
    for idx_rep = 1:1:nrep
        y = eta + randn(1, size(eta, 2))*sigma;
        b0 = [max(eta); mean(free)];
        [b_pred, ss, residual] = lsqcurvefit(@recbinding, b0, free, y); 
        bmax_pred(ctr_idx, idx_rep) = b_pred(1);
        kd_pred(ctr_idx, idx_rep) = b_pred(2);
    end
    
    % calculate theoretical sd
    p = 2;
    n = size(free, 2);
    yfit = residual + y;
    % predicted covbariance matrix is k(X'WX)^-1 where k = s/(n-p)
    k = ss/(n-p);
    % create X - partial exponetial partial
    xmat = zeros(n,p);
    delta =.001;
    beta = b_pred;
    for j = 1:p
        beta1 = beta;
        beta1(j) = (1+delta)*beta(j);
        y1 = recbinding(beta1, free);
        xmat(:,j) = (y1-yfit)./(delta*beta(j));
    end
    bcov = inv(xmat'*xmat)*k;
    bpredsd = (diag(bcov).^0.5)';
    bmax_sd_pred(ctr_idx) = bpredsd(1);
    kd_sd_pred(ctr_idx) = bpredsd(2);
end

%% calculate stats and plot
for i = 1:1:n_center
    bmax_sd(i) = std(bmax_pred(i,:));
    kd_sd(i) = std(kd_pred(i,:));
end
figure;
plot(log(center_list), log(bmax_sd));
hold on
plot(log(center_list), log(bmax_sd_pred));
legend('sample bmax sd','theoretical bmax sd');
xlabel('log center');
ylabel('log sd');

figure;
plot(log(center_list), log(kd_sd));
hold on
plot(log(center_list), log(kd_sd_pred));
legend('sample kd sd','theoretical kd sd');
xlabel('log center');
ylabel('log sd');