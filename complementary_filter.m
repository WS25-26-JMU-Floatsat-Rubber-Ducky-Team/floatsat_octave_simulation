addpath("utils");

%% ---------------- Complementary Filter Logic ---------------
function [roll_f, pitch_f, yaw_f, qf] = comp_step(roll_prev, pitch_prev, yaw_prev, gyr, acc, params)
  % Single complementary-filter update step performing:
  %  - integrate gyro to get angle prediction
  %  - compute accel-based roll/pitch
  %  - blend with ALPHA
  %  - return fused euler and quaternion (normalized)
  DT = params.DT; ALPHA = params.ALPHA; EPS = params.EPS;
  ix = params.IDX_X; iy = params.IDX_Y; iz = params.IDX_Z;

  % 1) Gyro integration (predict)
  roll_gyro  = roll_prev  + gyr(ix) * DT;
  pitch_gyro = pitch_prev + gyr(iy) * DT;
  yaw_gyro   = yaw_prev   + gyr(iz) * DT;   % needs magnetometer for correction

  % 2) Accelerometer-based angles (measurement)
  ax = acc(ix); ay = acc(iy); az = acc(iz);
  % Protect sqrt against tiny negative due to roundoff with EPS
  denom = sqrt(ay*ay + az*az + EPS);
  roll_acc  = atan2(ay, az);
  pitch_acc = atan2(-ax, denom);

  % 3) Complementary fusion (low-pass accel, high-pass gyro)
  roll_f  = ALPHA * roll_gyro  + (1 - ALPHA) * roll_acc;
  pitch_f = ALPHA * pitch_gyro + (1 - ALPHA) * pitch_acc;
  yaw_f   = yaw_gyro;                       % without mag: only integrate gyro

  % 4) Wrap angles to [-pi, pi] (helps numerical stability and porting)
  roll_f  = wrap_angle(roll_f);
  pitch_f = wrap_angle(pitch_f);
  yaw_f   = wrap_angle(yaw_f);

  % 5) Convert to quaternion and normalize
  qf = eul2quat(roll_f, pitch_f, yaw_f);
  qf = normalize_quat(qf);
endfunction

%% ---------------- Configuration / Constants ----------------
% Simulation parameters
SAMPLE_RATE_HZ = 100;            % samples per second
DT = 1.0 / SAMPLE_RATE_HZ;       % seconds per sample
SIM_DURATION_S = 1.0;            % simulation length in seconds
N = round(SIM_DURATION_S * SAMPLE_RATE_HZ);

% Physical constants
G = 9.81;                        % gravity magnitude (m/s^2)
EPS = 1e-8;                      % small epsilon for numeric safety

% Sensor axis indices (helps when porting later)
IDX_X = 1; IDX_Y = 2; IDX_Z = 3;

% Complementary filter tuning (use time-constant TAU, not a magic alpha)
% TAU is the time constant (seconds) of the low-pass applied to accel-based angle.
TAU = 0.5;                       % seconds (try 0.5..2.0)
ALPHA = TAU / (TAU + DT);        % alpha in [0,1] computed from TAU

% Initial yaw used when no magnetometer exists
INITIAL_YAW = 0.0;               % radians

%% ---------------- Simulated sensors ----------------
t = linspace(0, (N-1)*DT, N)';

% Gyroscope (rad/s). Simulated constant rotation about Y axis.
gyr = zeros(N,3);
gyr(:,IDX_Y) = 0.1;              % rad/s

% Accelerometer: gravity pointing along +Z (body frame)
acc = repmat([0.0, 0.0, G], N, 1);

%% ---------------- Preallocate states ----------------
roll  = zeros(N,1);
pitch = zeros(N,1);
yaw   = zeros(N,1);
Q     = zeros(N,4);              % quaternion [w x y z]

% Pack params for passing to functions (good for C struct later)
params.DT = DT;
params.ALPHA = ALPHA;
params.EPS = EPS;
params.IDX_X = IDX_X;
params.IDX_Y = IDX_Y;
params.IDX_Z = IDX_Z;

%% ---------------- Initial orientation from accelerometer ----------------
ax = acc(1,IDX_X); ay = acc(1,IDX_Y); az = acc(1,IDX_Z);
roll(1)  = atan2(ay, az);                                % roll from accel
pitch(1) = atan2(-ax, sqrt(ay^2 + az^2 + EPS));          % pitch from accel (safe denom)
yaw(1)   = INITIAL_YAW;

Q(1,:) = eul2quat(roll(1), pitch(1), yaw(1));
Q(1,:) = normalize_quat(Q(1,:));

%% ---------------- Complementary filter loop (encapsulated) ----------------
for k = 2:N
  % Read sensors (current sample)
  gyr_k = gyr(k, :);     % [gx gy gz] rad/s
  acc_k = acc(k, :);     % [ax ay az] m/s^2

  % one filter step (returns fused euler and quaternion)
  [roll(k), pitch(k), yaw(k), Q(k,:)] = comp_step(roll(k-1), pitch(k-1), yaw(k-1), ...
                                                  gyr_k, acc_k, params);
endfor

%% ---------------- Output ----------------
disp('Quaternions [w x y z] (all samples):');
disp(Q);
