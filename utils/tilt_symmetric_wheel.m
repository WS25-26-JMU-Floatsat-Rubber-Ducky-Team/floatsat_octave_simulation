function [sigma, cond_num, N] = tilt_symmetric_wheel(tilt_deg)
  % Computes wheel axes, singular values, and condition number
  % tilt_deg : tilt angle in degrees from +Z axis

  % Convert tilt to radians
  tilt = deg2rad(tilt_deg);

  % Azimuth angles for symmetric 3-wheel layout
  az = deg2rad([0, 120, 240]);

  % Preallocate 3x3 axis matrix
  N = zeros(3,3);

  for i = 1:3
    nx = sin(tilt) * cos(az(i));
    ny = sin(tilt) * sin(az(i));
    nz = cos(tilt);

    n = [nx; ny; nz];
    n = n / norm(n);   % safe normalization
    N(:, i) = n;
  endfor

  % Singular values
  sigma = svd(N);

  % Condition number
  cond_num = max(sigma) / min(sigma);
endfunction
