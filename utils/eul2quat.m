function q = eul2quat(roll, pitch, yaw)
  % Convert ZYX (yaw-pitch-roll) Euler to quaternion [w x y z]
  cr = cos(roll/2); sr = sin(roll/2);
  cp = cos(pitch/2); sp = sin(pitch/2);
  cy = cos(yaw/2); sy = sin(yaw/2);

  w = cr*cp*cy + sr*sp*sy;
  x = sr*cp*cy - cr*sp*sy;
  y = cr*sp*cy + sr*cp*sy;
  z = cr*cp*sy - sr*sp*cy;
  q = [w x y z];
endfunction
