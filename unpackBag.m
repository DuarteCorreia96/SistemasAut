bag_name = 'pinto';
bag_location = strcat('D:\bags\',bag_name,'.bag');
bag = rosbag(bag_location);

odom_bag = select(bag, 'Topic', '/pioneer/pose');
mocap_bag = select(bag, 'Topic', '/mocap_node/Robot_1/ground_pose');
aruco_bag = select(bag, 'Topic','/aruco_marker_publisher/markers');

odom = timeseries(odom_bag, 'Twist.Twist.Linear.X', ...
                           'Twist.Twist.Angular.Z');
                       
mocap = timeseries(mocap_bag, 'X', 'Y');

aruco_msgs = readMessages(aruco_bag,'DataFormat','struct');

L = length(aruco_msgs);
M = 2*L;
aruco = zeros(M,4);

m = 1;

for j = 1:L
    
    msg = aruco_msgs{j,1};
    markers = msg.Markers;
    
    K = length(markers);
    
    if K == 0
        continue
    end
    
    time = msg.Header.Stamp;
    aruco(m:m+K-1,1) = int64(time.Sec)*10^9 + int64(time.Nsec);
    
    for k = 1:K
        aruco(m+k-1,2) = markers(k).Id;
        aruco(m+k-1,3) = markers(k).Pose.Pose.Position.X;
        aruco(m+k-1,4) = markers(k).Pose.Pose.Position.Z;
    end
    
    m = m + K;    
end

aruco(~any(aruco,2),:) = [];

% index_m = find(mocap > 5 || mocap < T);
% index_o = find(odom > 5 || odom < T);
% odom = delsample(odom,'Index',index_o);
% mocap = delsmaple(mocap, 'Index', index_m);

% table = readtable('data.txt');
% aruco_raw = table2array(table);
% aruco_raw(:,1) = aruco_raw(:,1) - aruco_raw(1,1);
% aruco_raw(:,1) = aruco_raw(:,1) + aruco_bag.StartTime;

odom.Time = odom.Time - bag.StartTime;
mocap.Time = mocap.Time - bag.StartTime;
aruco(:,1) = aruco(:,1)*1e-9 - bag.StartTime;

% Numerical precision errors...
aruco(aruco(:,1) < 0, 1) = 0;
