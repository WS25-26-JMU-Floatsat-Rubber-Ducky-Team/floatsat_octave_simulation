% Symmetric tilted reaction wheel cluster
% Each wheel axis is a unit vector in body coordinates.

% Tilt angle relative to +Z axis (vertical).
tilt = deg2rad(54.736);

% Angles around the Z-axis for the three wheels
az = deg2rad([0, 120, 240]);

% Preallocate axis matrix (3x3)
N = zeros(3,3);

for i = 1:3
  % Each wheel axis vector in spherical coordinates:
  %   n = [sin(tilt)*cos(az), sin(tilt)*sin(az), cos(tilt)]
  nx = sin(tilt) * cos(az(i));
  ny = sin(tilt) * sin(az(i));
  nz = cos(tilt);

  % Normalize
  n = [nx; ny; nz];
  n = n / norm(n);

  N(:, i) = n;
endfor

disp("Wheel axis matrix N (columns are axes):");
disp(N);

% Evaluate control authority (singular values)
sigma = svd(N);

disp("Singular values (torque authority):");
disp(sigma);

cond_num = max(sigma) / min(sigma);
disp(["Condition number: ", num2str(cond_num)]);

% Compute closest orthogonal matrix
[U, ~, V] = svd(N);
R = U * V';

disp("Closest orthogonal rotation matrix R:");
disp(R);
disp("Closest orthogonal rotation matrix R':");
disp(R');
