function ans = sum_abs_sine(n)
% calculate the sum of absolute sine values
i = 1:1:n;
sin_i = abs(sin(i));
ans = sum(sin_i);
end
