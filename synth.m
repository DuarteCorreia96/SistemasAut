clear classes

addpath('MATLAB_code')

pymod_ekf = py.importlib.import_module('ekf');
pymod_models = py.importlib.import_module('synth_base');
pymod_matlab = py.importlib.import_module('synth_matlab');

py.importlib.reload(pymod_ekf);
py.importlib.reload(pymod_models);
py.importlib.reload(pymod_matlab);

% Parâmetros do ekf
Q_diag = [0.06, 0.06];
sigma  = 0.01;

% Variâncias dos sensores
mu_odom = [0.01, 1];
mu_obse = [0.01, 0.02];

ekf = py.synth_matlab.Matlab_EKF(Q_diag, sigma, mu_odom, mu_obse);

% Valores máximos observação
max_angle    = pi / 6;
max_distance = 1000;

% Controlos a dar ao robot
v  = 0.09;
w  = 0.06;
dt = 0.01;

% Mudança nos controlos a cada periodo
w_var = 0.0001;
v_var = 0;

% Numero de ponto para a animação e quantos ciclos de ekf são skipped entre
% frames
skipped = 10;
npoints = 300;
xsize = [-.5 1.6];
ysize = [-.1 1.8];

% Inserir landmarks no simulador
ekf.add_landmark( 1, -0.5, 0.5)
ekf.add_landmark( 2,  0.5, 1.5)
% ekf.add_landmark( 3, .5, 1.2)
% ekf.add_landmark( 4,-.2, .1)
% ekf.add_landmark( 8,-.1, 1.2)
% ekf.add_landmark( 5, 1.0, .2)
% ekf.add_landmark( 6, 1.5, .3)
% ekf.add_landmark( 7, 1.2, 1.1)

% Não deve ser preciso mexer daqui para baixo
landmarks_x = np_matlab(ekf.get_landmarks_x());
landmarks_y = np_matlab(ekf.get_landmarks_y());

robot_path_x = zeros(npoints, 1);
robot_path_y = zeros(npoints, 1);

estim_path_x = zeros(npoints, 1);
estim_path_y = zeros(npoints, 1);

noise_path_x = zeros(npoints, 1);
noise_path_y = zeros(npoints, 1);

% fig = figure('units','normalized','outerposition',[0 0 1 1]);
fig = figure();
set(fig,'defaultLegendAutoUpdate','off')
hold on
grid minor
xlim(xsize)
ylim(ysize)

h = zeros(3, 1);
h(1) = plot(NaN, NaN, 'k.-');
h(2) = plot(NaN, NaN, 'b.-');
h(3) = scatter(NaN, NaN, 'ob');
h(4) = plot(NaN, NaN, 'y.-');
h(5) = scatter(NaN, NaN, 'xg');
h(6) = scatter(NaN, NaN, 'xr');
h(7) = scatter(NaN, NaN, 'or');
% legend(h, 'Robot','Estimate', 'Estimate Covariance' ,'Odom', ...
%     'Landmark', 'Landmark Estimation', 'Landmark Covariance', ...
%     'Location', 'Northeast');

title('EKF-SLAM with Known Correspondences using Synthetic Data')
xlabel('x [m]')
ylabel('y [m]')

for i = 1:npoints * skipped
    w = w + w_var;
    v = v + v_var;
    
    ekf.update_step(max_angle, max_distance)
    ekf.prediction_step(v, w, dt)

    if mod(i, skipped) == 0
        k = i / skipped;
        
        pause(0.0001)
        cla

        estimate   = np_matlab(ekf.get_estimate());
        covariance = np_matlab(ekf.get_covariance());

        scatter(landmarks_x, landmarks_y, 'kx')

        current_robot = np_matlab(ekf.get_robot());
        current_noise = np_matlab(ekf.get_odom());

        robot_path_x(k) = current_robot(1);
        robot_path_y(k) = current_robot(2);
        plot(robot_path_x(1:k), robot_path_y(1:k), 'k.-')
        
        noise_path_x(k) = current_noise(1);
        noise_path_y(k) = current_noise(2);
        plot(noise_path_x(1:k), noise_path_y(1:k), 'y.-')
        
        estim_path_x(k) = estimate(1);
        estim_path_y(k) = estimate(2);
        plot(estim_path_x(1:k), estim_path_y(1:k), 'blue.-')

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
end


