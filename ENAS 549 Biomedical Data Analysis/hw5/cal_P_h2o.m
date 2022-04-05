function ans=cal_P_h2o(T)
    ans = exp(9.214 - 1049.8/(1.985*(32+1.8*T)))-0.298;
end