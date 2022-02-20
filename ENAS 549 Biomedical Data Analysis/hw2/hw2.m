mu = 100;
sigma = 5;

%%
x = 100;
y = cdf('Normal',x,mu,sigma);
1-y

%%
x = 95;
y = cdf('Normal',x,mu,sigma);
y

%%
x1 = 90;
x2 = 110;
y1 = cdf('Normal',x1,mu,sigma);
y2 = cdf('Normal',x2,mu,sigma);
y2-y1

%%
x = 108.225;
y = cdf('Normal',x,mu,sigma);
y

%% 
clear all
n = 1000;
mu = 0;
% sigma = 0.5;
% sigma = 1;
sigma = 1.5;
x = normrnd(mu, sigma, n, 1);
y = cos(x);
mean(y)