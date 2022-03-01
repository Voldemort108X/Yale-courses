% s = 0:1:3;
% q = zeros((size(s,2)),1);
clear all;
cnt = 1;
max_val = 1000;
for s=0:1:max_val
%     fun = @(y) -1/(2*sqrt(2*pi*s))*(exp(-(y-1).^2/(2*s))+exp(-(y+1).^2/(2*s))).*(1./(1+exp(2*y/s)).*log2(1./(1+exp(2*y/s)))+1./(1+exp(-2*y/s)).*log2(1./(1+exp(-2*y/s))));
    fun = @(y) -1/(2*sqrt(2*pi*s))*(exp(-(y-1).^2/(2*s))+exp(-(y+1).^2/(2*s))).*log2(1/(2*sqrt(2*pi*s))*(exp(-(y-1).^2/(2*s))+exp(-(y+1).^2/(2*s))));
    q(cnt) = integral(fun, -Inf, Inf);
    cnt = cnt + 1;
end

s = 0:1:max_val;
plot(s,q);
xlabel('s');
ylabel('f(s)');
figure;
plot(s, 1/2*log2(2*pi*exp(1)*s)-q);

% s = 190;
% fun = @(y)
% 1/(2*sqrt(2*pi*s))*(exp(-(y-1).^2/(2*s))+exp(-(y+1).^2/(2*s))); % sqrt
% (s) is super super important!!!!
% q = integral(fun, -Inf, Inf);