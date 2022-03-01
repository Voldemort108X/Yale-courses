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

%%
t = 0:1:60;
A1=50;
b1=0.1;
A2=50;
b2=0.01;

c = A1*exp(-b1*t)+A2*exp(-b2*t);
noise = normrnd(0, 5, 1, size(c,2));
d = c + noise;
figure;
plot(t, c);
hold on
plot(t, d);
legend('c', 'd');

S = zeros(1000,1);
for i=1:1:1000
    c = A1*exp(-b1*t)+A2*exp(-b2*t);
    noise = normrnd(0, 5, 1, size(c,2));
    d = c + noise;
    S(i) = sum((d-c).^2);
end

figure;
hist(S);