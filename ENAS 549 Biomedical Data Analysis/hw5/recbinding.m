%recbinding.m
% calculate b(1) x / (b(2)+x)
% b is parameter vector (Bmax, Kd)

function F=recbinding(b,x)
    F=b(1)*x./(b(2)+x);
