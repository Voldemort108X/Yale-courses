%exponentials
% calculate A(1) exp(-alpha(1)t) + A(2) exp(-alpha(2)t)...
% b is parameter vector (A1,alpha1,A2,alpha2,...)
% if length of b is odd, assume alphan=0
function F=exponentials(b,t)

F=0.*t;
nexp=fix((length(b)+1)/2);
A=b(1:2:length(b));
alpha=b(2:2:length(b));
if length(alpha) < nexp 
       alpha=[alpha,0];
end
for i=1:nexp
    F=F+A(i)*exp(-alpha(i)*t);
end
