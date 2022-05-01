function [alpha_n,beta_n,alpha_m,beta_m,alpha_h,beta_h]=rate_constants(v)
% rate_constants.m
% Calculates the rate constants for the Hodgkin-Huxley model

alpha_n=0.01*(10-v)./(exp((10-v)/10)-1);
beta_n=0.125*exp(-v/80);
alpha_m=0.1*(25-v)./(exp((25-v)/10)-1);
beta_m=4*exp(-v/18);
alpha_h=0.07*exp(-v/20);
beta_h=1./(exp((30-v)/10)+1);
