clear
close all
K = 10000;
v = zeros(3,K);
v_true = zeros(3,K);
h = 0.01;

sigma = 0.01;
cmd = [0.1 0.02];
cov_m = zeros(3,3,K);


for j = 2:K
    e = genNoise(sigma);
    v_true(:,j) = predict_mean(v_true(:,j-1),cmd,h);
    v(:,j) = predict_mean(v(:,j-1),cmd+e,h);
    [Gx,Ge] = getG(v(:,j-1),cmd,h);
    cov_m(:,:,j) = predict_cov(cov_m(:,:,j-1),sigma,Gx,Ge);
end

cov_xy = cov_m(1:2,1:2,:);

n = 3:K/5:K;

figure
for j = 1:length(n)
    h = error_ellipse(cov_xy(:,:,n(j)),v(1:2,n(j)));
    
    if mod(j,2)
        h.Color = [0 1 0];
    else
        h.Color = [1 0 1];
    end
    
    hold on
    plot(v(1,n(j)),v(2,n(j)),'rx','MarkerSize',10);
    hold on
end

hAx1 = plot(v_true(1,:),v_true(2,:),'Color','r');
hold on
hAx2 = plot(v(1,:),v(2,:),'Color','b');
grid minor

xlabel('$x$ [m]','Interpreter','latex','Fontsize',12);
ylabel('$y$ [m]','Interpreter','latex','Fontsize',12);
a_title = 'Covariance matrix propagation';
title(a_title,'Interpreter','latex','Fontsize',12)
legend([hAx1 hAx2], {'Real','Observed'},'location','best','Interpreter',...
                                                                   'latex')


error = (v - v_true)*1000;

figure
subplot(2,1,1)
plot(1:K,error(1,:))
title('Error in $x$ evolution','Interpreter','latex','Fontsize',12)
ylabel('$x$ [mm]','Interpreter','latex','Fontsize',12);
grid minor
subplot(2,1,2)
plot(1:K,error(2,:))
title('Error in $y$ evolution','Interpreter','latex','Fontsize',12)
ylabel('$y$ [mm]','Interpreter','latex','Fontsize',12);
grid minor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e = genNoise(sigma)
    R = sigma^2*eye(2);
    e = randn(1,2)*chol(R);
end
