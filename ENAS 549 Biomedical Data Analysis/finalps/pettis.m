function dy=pettis(t,y,extra)
% pet tissue model, one or 2 compartment
% extra holds input function
k=extra.k;
cp=interp1(extra.t_cp,extra.cp,t);
nc=length(y);
if nc == 1 
    dy=k(1)*cp-k(2)*y;
else
    dy=[k(1)*cp-(k(2)+k(3))*y(1)+k(4)*y(2)
        k(3)*y(1)-k(4)*y(2)];
end

return
end
