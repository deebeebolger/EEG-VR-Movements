function accel_out = accel_calc(velIn, T)

veldiff = diff(velIn);
dT = diff(T);
accel_out = veldiff./dT(2:length(velIn));

end