%% problem 1

%% newton's method
T = 15;
epsilon = 0.0001;
nmax = 1000;
err = inf;
cnt = 1;
value_p = 0.298;
while err>=epsilon && cnt<= nmax
    y = exp(9.214 - 1049.8/(1.985*(32+1.8*T)))-value_p;
    dy = (2.94892*10^6*exp(-293.815/(17.7778 + T)))/(17.7778 + T)^2;
    T = T - y/dy;
    err = abs(y);
    cnt = cnt + 1;
end


%% bisection method
intv = [5, 20];
epsilon = 0.0001;
nmax = 1000;
err = inf;
cnt = 1;
value_p = 0.298;
while abs(err)>=epsilon && cnt<=nmax
    x_pred = mean(intv);
    err = exp(9.214 - 1049.8/(1.985*(32+1.8*x_pred)))-value_p;
    if err>=0
        intv(2) = x_pred;
    else
        intv(1) = x_pred;
    end
    cnt = cnt + 1;
end

