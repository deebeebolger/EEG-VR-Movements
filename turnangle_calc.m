function [turnangle_out,trs] = turnangle_calc(dataIn,T)

[posx_rs,trs] = resample(dataIn(:,1),T,8,10,1);
[posy_rs,trs] = resample(dataIn(:,2),T,8,10,1);
[posz_rs,trs] = resample(dataIn(:,3),T,8,10,1);

dataIn_rs = [posx_rs,posy_rs];

thetadeg1 = zeros(size(dataIn_rs,1)-2,1);
thetadeg2 = zeros(size(dataIn_rs,1)-2,1);
costheta = zeros(size(dataIn_rs,1)-2,1);
costheta2 = zeros(size(dataIn_rs,1)-2,1);


for cnt = 2:size(dataIn_rs,1)-1
    
  numer1 = dot(dataIn_rs(cnt,:),dataIn_rs(cnt-1,:));
  denom1 = norm(dataIn_rs(cnt,:))*norm(dataIn_rs(cnt-1,:));
  costheta(cnt-1) = numer1/denom1;
  td1 = acosd(costheta(cnt-1));
  thetadeg1(cnt-1) =  td1;      %express in degrees
  
  numer2 = dot(dataIn_rs(cnt,:),dataIn_rs(cnt+1,:));
  denom2 = norm(dataIn_rs(cnt,:))*norm(dataIn_rs(cnt+1,:));
  costheta2(cnt-1) = numer2/denom2;
  td2 = acosd(costheta2(cnt-1));
  thetadeg2(cnt-1) = td2;        %express in degrees
    
end

turnangle_out = [thetadeg1,thetadeg2];


end