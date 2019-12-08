bag = rosbag('duarte.bag');

odom_bag  = select(bag, 'Topic', '/pioneer/pose');
mocap_bag = select(bag, 'Topic', '/mocap_node/Robot_1/ground_pose');
aruco_bag = select(bag, 'Topic', '/aruco_marker_publisher/markers');

odom = timeseries(odom_bag, 'Pose.Pose.Position.X', 'Twist.Twist.Angular.Z');
odom.Time = odom.Time - bag.StartTime;

mocap = timeseries(mocap_bag, 'X', 'Y');
mocap.Time = mocap.Time - bag.StartTime;


aruco = timeseries(aruco_bag, 'MarkerArray.Header');
aruco.Time = aruco.Time - bag.StartTime;