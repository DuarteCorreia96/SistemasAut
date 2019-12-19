% clear classes
% load('bag_treated.mat')
addpath('MATLAB_code')

pymod_ekf = py.importlib.import_module('ekf');
pymod_matlab = py.importlib.import_module('real_matlab');

py.importlib.reload(pymod_ekf);
py.importlib.reload(pymod_matlab);

% Parâmetros do ekf
Q_diag = [0.06, 0.5];
sigma  = 0.1;
L = 1;

ekf = py.real_matlab.Matlab_EKF(Q_diag, sigma, L);

% Numero de ponto para a animação e quantos ciclos de ekf são skipped entre
% frames
xsize = [-7 7];
ysize = [-7 7];

% Não deve ser preciso mexer daqui para baixo
% estim_path_x = zeros(npoints, 1);
% estim_path_y = zeros(npoints, 1);
% 
% noise_path_x = zeros(npoints, 1);
% noise_path_y = zeros(npoints, 1);

robot_path_x = [];
robot_path_y = [];

% MoCap Frame to World Frame!
landmarks_x = [0.68 1.35 2.53 3.57] + mocap.Data(1,1);
landmarks_y = [-0.12 0.62 0.67 -0.35] + mocap.Data(1,2);

fig = figure('units','normalized','outerposition',[0 0 1 1]);
set(fig,'defaultLegendAutoUpdate','off')
hold on
box on
grid minor
xlim(xsize)
ylim(ysize)

h = zeros(3, 1);
h(1) = plot(NaN, NaN, 'g.-');
h(2) = plot(NaN, NaN, 'b.-');
h(3) = scatter(NaN, NaN, 'ob');
h(4) = plot(NaN, NaN, 'r.-');
h(5) = scatter(NaN, NaN, 'xg');
h(6) = scatter(NaN, NaN, 'xr');
h(7) = scatter(NaN, NaN, 'or');
legend(h, 'Robot','Estimate', 'Estimate Covariance' ,'Odom', ...
    'Landmark', 'Landmark Estimation', 'Landmark Covariance', ...
    'Location', 'Northeast');

title('EKF Real Data')
xlabel('x [m]')
ylabel('y [m]')

time_arr = [odom.Time; mocap.Time; aruco(:,1)];
% time_arr = [odom.Time; mocap.Time];
max_time = max(time_arr);

k_mocap = 1;
k_aruco = 1;
k_odom  = 1;
cur_time = 0;

while cur_time < max_time
    disp(cur_time);
    
    if k_aruco < length(aruco(:,1)) ...
        && (aruco(k_aruco, 1) - cur_time) < (mocap.Time(k_mocap) - cur_time) ...
        && (aruco(k_aruco, 1) - cur_time) < (odom.Time(k_odom) - cur_time)
       
        id    = aruco(k_aruco, 2);
        r     = sqrt(aruco(k_aruco, 3)^2 + aruco(k_aruco, 4)^2);
        theta = -atan2(aruco(k_aruco, 3), aruco(k_aruco, 4));
        
        ekf.update_step(id, r, theta)
    
        cur_time = aruco(k_aruco, 1);
        k_aruco  = k_aruco + 1;
    
    elseif k_mocap < length(mocap.Time) ...
        && (mocap.Time(k_mocap) - cur_time) < (odom.Time(k_odom) - cur_time) ...
        && (mocap.Time(k_mocap) - cur_time) < (aruco(k_aruco, 1) - cur_time)
       
        robot_path_x = -(mocap.Data(1:k_mocap, 1) - mocap.Data(1, 1));
        robot_path_y = -(mocap.Data(1:k_mocap, 2) - mocap.Data(1, 2));
    
        cur_time = mocap.Time(k_mocap);
        k_mocap  = k_mocap + 10;
    
    elseif k_odom < length(odom.Time) ...
            && (odom.Time(k_odom) - cur_time) < (mocap.Time(k_mocap) - cur_time) ...
            && (odom.Time(k_odom) - cur_time) < (aruco(k_aruco, 1) - cur_time)
       
        v  = odom.Data(k_odom, 1);
        w  = odom.Data(k_odom, 2);
        
        if k_odom == 1
            dt = odom.Time(1);
        else
            dt = odom.Time(k_odom) - odom.Time(k_odom-1);
        end
        
        if (abs(v)) > 0.01 || (abs(w)) > 1e-3
            ekf.prediction_step(v, w, dt)
        end        
        
        cur_time = odom.Time(k_odom);
        k_odom  = k_odom + 1;
    end
    
    pause(0.0001)
    cla

    estimate   = np_matlab(ekf.get_estimate());
    covariance = np_matlab(ekf.get_covariance());

    scatter(landmarks_x, landmarks_y, 'gx')

    plot(robot_path_x, robot_path_y, 'g.-')

    estim_path_x(k_odom) = estimate(1);
    estim_path_y(k_odom) = estimate(2);
    plot(estim_path_x(1:k_odom), estim_path_y(1:k_odom), 'blue.-')

    try 
        h = error_ellipse(covariance(1:2,1:2), estimate(1:2));
        h.Color = 'Blue';
        plot(h)
    catch
    end

    for j = 0: (length(covariance) - 3) / 2 - 1

        x = 4 + j * 2;
        y = 5 + j * 2;
        scatter(estimate(x),estimate(y), 'rx')
        try 
            h = error_ellipse(covariance(x:y,x:y), estimate(x:y));
            h.Color = 'Red';
            plot(h)
        catch
        end
    end
end


