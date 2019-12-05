% clear classes
% 
% mod = py.importlib.import_module('synth_matlab');
% py.importlib.reload(mod);

Q_diag = [5, 5];
sigma  = 0.02;

mu_odom = [0.2, 0.05];
mu_obse = [0.5, 0.02];

ekf = py.synth_matlab.Matlab_EKF(Q_diag, sigma, mu_odom, mu_obse);

v  = 0.2;
w  = 0.02;
dt = 0.01;

ekf.add_landmark( 1, 2, 3)
ekf.add_landmark( 2, 3, 3)
ekf.add_landmark( 3, 0, 4)
ekf.add_landmark( 4, 2, 4)

ekf.update_step()
ekf.prediction_step(v, w, dt)

estimate   = np_matlab(ekf.get_estimate());
covariance = np_matlab(ekf.get_covariance());