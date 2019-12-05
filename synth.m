clear classes

pymod_matlab = py.importlib.import_module('synth_matlab');
pymod_ekf    = py.importlib.import_module('ekf');
pymod_models = py.importlib.import_module('synth_base');

py.importlib.reload(pymod_matlab);
py.importlib.reload(pymod_ekf);
py.importlib.reload(pymod_models);

% Par�metros do ekf
Q_diag = [0.1, 0.1];
sigma  = 0.02;

% Vari�ncias dos sensores
mu_odom = [0.1, 0.05];
mu_obse = [0.2, 0.08];

ekf = py.synth_matlab.Matlab_EKF(Q_diag, sigma, mu_odom, mu_obse);

% Controlos a dar ao robot
v  = 0.9;
w  = 0.1;
dt = 0.01;

% Mudan�a nos controlos a cada periodo
w_var = 0.0001;
v_var = 0;

% Numero de ponto para a anima��o e quantos ciclos de ekf s�o skipped entre
% frames
skipped = 10;
npoints = 600;
xsize = [-4 11];
ysize = [-1 15];

% Inserir landmarks no simulador
ekf.add_landmark( 1, 5, 7)
ekf.add_landmark( 3,-2, 10)
ekf.add_landmark( 4, 8, 12)
ekf.add_landmark( 5,-2, 1)

% N�o deve ser preciso mexer daqui para baixo
landmarks_x = np_matlab(ekf.get_landmarks_x());
landmarks_y = np_matlab(ekf.get_landmarks_y());

robot_path_x = zeros(npoints, 1);
robot_path_y = zeros(npoints, 1);

estim_path_x = zeros(npoints, 1);
estim_path_y = zeros(npoints, 1);

noise_path_x = zeros(npoints, 1);
noise_path_y = zeros(npoints, 1);

figure('units','normalized','outerposition',[0 0 1 1])
for i = 1:npoints * skipped
    w = w + w_var;
    v = v + v_var;
    
    ekf.update_step()
    ekf.prediction_step(v, w, dt)

    if mod(i, skipped) == 0
        k = i / skipped;
        
        pause(0.001)
        clf

        estimate   = np_matlab(ekf.get_estimate());
        covariance = np_matlab(ekf.get_covariance());

        scatter(landmarks_x, landmarks_y, 'gx')
        hold on
        grid minor
        xlim(xsize)
        ylim(ysize)

        current_robot = np_matlab(ekf.get_robot());
        current_noise = np_matlab(ekf.get_odom());

        robot_path_x(k) = current_robot(1);
        robot_path_y(k) = current_robot(2);
        plot(robot_path_x(1:k), robot_path_y(1:k), 'g.-')
        hold on
        
        noise_path_x(k) = current_noise(1);
        noise_path_y(k) = current_noise(2);
        plot(noise_path_x(1:k), noise_path_y(1:k), 'r.-')
        hold on
        
        estim_path_x(k) = estimate(1);
        estim_path_y(k) = estimate(2);
        plot(estim_path_x(1:k), estim_path_y(1:k), 'blue.-')
        hold on

        try 
            h = error_ellipse(covariance(1:2,1:2), estimate(1:2));
            h.Color = 'Blue';
            plot(h)
            hold on
        catch
        end
        
        for j = 0: (length(covariance) - 3) / 2 - 1

            x = 4 + j * 2;
            y = 5 + j * 2;
            scatter(estimate(x),estimate(y), 'rx')
            hold on
            try 
                h = error_ellipse(covariance(x:y,x:y), estimate(x:y));
                h.Color = 'Red';
                plot(h)
                hold on
            catch
            end
        end
    end
end


