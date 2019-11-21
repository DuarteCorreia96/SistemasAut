function x_v = predict_mean(mu_old,odom,h)
        
    x_old = mu_old(1);
    y_old = mu_old(2);
    th_old = mu_old(3);
    
    v = odom(1);
    w = odom(2);
        
    x = x_old + v*cos(th_old)*h;
    y = y_old + v*sin(th_old)*h;
    th = th_old + w*h;
    
    x_v = [x, y, th]';
end

