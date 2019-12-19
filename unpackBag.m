bag = rosbag('pinto.bag');
odom_bag = select(bag, 'Topic', '/pioneer/pose');
mocap_bag = select(bag, 'Topic', '/mocap_node/Robot_1/ground_pose');

aruco_bag = select(bag, 'Topic','/aruco_marker_publisher/markers');

odom = timeseries(odom_bag, 'Twist.Twist.Linear.X', ...
                           'Twist.Twist.Angular.Z');
                       
mocap = timeseries(mocap_bag, 'X', 'Y');

% index_m = find(mocap > 5 || mocap < T);
% index_o = find(odom > 5 || odom < T);
% odom = delsample(odom,'Index',index_o);
% mocap = delsmaple(mocap, 'Index', index_m);

table = readtable('data.txt');
aruco_raw = table2array(table);
aruco_raw(:,1) = aruco_raw(:,1) - aruco_raw(1,1);
aruco_raw(:,1) = aruco_raw(:,1) + aruco_bag.StartTime;

odom.Time = odom.Time - bag.StartTime;
mocap.Time = mocap.Time - bag.StartTime;
aruco_raw(:,1) = aruco_raw(:,1) - bag.StartTime;
