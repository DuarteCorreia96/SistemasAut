function [Gx,Ge] = getG(mu_old,cmd,h)
    
    th_old = mu_old(3);
    L = 0.25;
%     L = 3;
    
%     e_l = e(1);
%     e_r = e(2);
    
    Gx = eye(3);
    
    v = cmd(1);
    c = cos(th_old);
    s = sin(th_old);
    
    B = zeros(3);
%     B(1:2,3) = [-(v + (e_r + e_l)/2)*s*h; (v + (e_r + e_l)/2)*c*h];
    B(1:2,3) = [-v*s*h; v*c*h];
    
    Gx = Gx + B;
    
    Ge = [c c;s s;1/L -1/L];
     
end

