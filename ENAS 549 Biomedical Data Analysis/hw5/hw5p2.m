%% problem 2
beta_true = [50, 0.01, 0.1];
b0 = [50, 0.01, 0.1];
sigma = 1;

nrep = 100;

for iter=1:1:nrep
    b_est = gaussfit_hw(beta_true, b0, sigma);
    b1_est(iter) = b_est(1);
    b2_est(iter) = b_est(2);
    b3_est(iter) = b_est(3);
end

b1_mean = mean(b1_est);
b1_std = std(b1_est);
b2_mean = mean(b2_est);
b2_std = std(b2_est);
b3_mean = mean(b3_est);
b3_std = std(b3_est);

corr_b1b2 = corrcoef(b1_est, b2_est);
corr_b1b3 = corrcoef(b1_est, b3_est);
corr_b2b3 = corrcoef(b2_est, b3_est);

figure;
scatter(b1_est, b2_est);
xlabel('b1 est');
ylabel('b2 est');

figure;
scatter(b1_est, b3_est);
xlabel('b1 est');
ylabel('b3 est');

figure;
scatter(b2_est, b3_est);
xlabel('b2 est');
ylabel('b3 est');

function b=gaussfit_hw(beta,b0,percsd)
 %beta=[1,1,.1];
 %b0=[.5,.5,.05];
 %percsd=10;
 display(beta);
 %create the data
 % skip t=0 since 0^x not defined
 t = (1:5:100)';  %column vector
 yper = beta(1)*(exp(-beta(2)*t) - exp(-beta(3)*t));
 n = length(t);
 p = 3;
 % simulate noisy data with percsd % error
 y=yper + randn(n,1) * percsd;
 eps=.001;    %fractional change allowed
 b=b0;   % starting guess
 maxchg=1.0;
 maxiter=100;
 iter=0;
 display(b);
 eta1 = b(1)*(exp(-b(2)*t) - exp(-b(3)*t));
 while ((maxchg > eps) & (iter < maxiter))
     iter=iter+1;
     % b(n+1)=b(n)+ inv(X'X)X'(y-eta(n))
     % current function evaluation
     eta = b(1)*(exp(-b(2)*t) - exp(-b(3)*t));
     ss=sum((y-eta).^2);
     display(['Iteration ',int2str(iter),' Sum of squares is ',num2str(ss)]);
     % analytical derivatives
     detadb1 = exp(-b(2)*t) - exp(-b(3)*t);
     detadb2 = -t.*(b(1)*exp(-b(2)*t));
     detadb3 = t.*(b(1)*exp(-b(3)*t));
     xmat=zeros(n,p);
     xmat(:,1)=detadb1;
     xmat(:,2)=detadb2;
     xmat(:,3)=detadb3;
     delta_b=inv(xmat'*xmat)*xmat'*(y-eta);
     maxchg=max(abs(delta_b'./b));
     b=b+delta_b';
     display(b);
 end
 plot(t,yper,'-',t,y,'x',t,eta,'r',t,eta1,'g');
 ylim([0,max(eta)]);
end

