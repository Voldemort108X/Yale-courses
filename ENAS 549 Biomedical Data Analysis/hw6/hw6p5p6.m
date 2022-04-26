%% problem 5
x = [ones(55,1); -ones(21,1); zeros(2,1)];
y = zeros(78,1);

% with signtest function
[p, h, stats] = signtest(x,y);

% without signtest function
z = (21 - (78-2)/2)/(sqrt((78-2)/4));
p_manual = 2*cdf('Normal',z,0,1);

%% problem 6
func1 = @(x,t) x(1)*exp(-x(2)*t);
func2 = @(x,t) x(1)*exp(-x(2)*t) + x(3);

p1 = 2;
p2 = 3;
n = 40;

n_rep = 500;
t = 1:1:40;
beta1 = 20;
beta2 = 0.1;
sigma = 2;

cnt_success = 0;
for i = 1:1:n_rep
    ydata = exponentials([beta1, beta2], t) + sigma * randn(1, size(t,2));
    
    x10 = [beta1, beta2];
    x20 = [beta1, beta2, 0];
    
    x1_hat = lsqcurvefit(func1, x10, t, ydata);
    x2_hat = lsqcurvefit(func2, x20, t, ydata);
    
    y1_hat = exponentials(x1_hat, t);
    y2_hat = exponentials(x2_hat, t);
    
    ss1 = sum((y1_hat-ydata).^2);
    ss2 = sum((y2_hat-ydata).^2);
    
    F = ((ss1-ss2)/(p2-p1))/(ss2/(n-p2));
    pF = fcdf(F, p2-p1, n-p2);
    
    if pF > 0.95
        cnt_success = cnt_success + 1;
    end
   
end