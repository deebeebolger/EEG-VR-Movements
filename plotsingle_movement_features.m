function plotsingle_movement_features(hdl,~)

D=get(hdl,'UserData');
vel = D{1,1};
accel = D{1,2};
tangle = D{1,3};
time = D{1,4};
figtitle = D{1,5};
time_rs = D{1,6};

figure('Name',figtitle);
subplot(1,3,1)
plot(time_rs(1:end-1),vel)
title('Velocity')

subplot(1,3,2)
plot(time_rs(2:end-1),accel)
title('Acceleration')

subplot(1,3,3)
plot(time_rs(2:end-1),tangle,'r-o')
title('Turning Angle (degrees)');

end