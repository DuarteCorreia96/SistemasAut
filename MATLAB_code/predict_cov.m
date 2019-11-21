function cov = predict_cov(cov_old,sigma,Gx,Ge)
    
    R = sigma^2*(Ge*Ge');
    cov = Gx*cov_old*Gx' + R;
    
end

