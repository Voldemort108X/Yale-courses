function ans=approx_pi(n) 
% calculate pi with Wallis series  
% pi/2= product of (4n^2/(2n-1)/(2n+1))  for n terms 
ans=1; 
for k=1:n 
    ans=ans*4*k^2/(2*k-1)/(2*k+1); 
end 
ans=2*ans; 
end 