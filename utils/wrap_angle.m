function a = wrap_angle(a)
  % Wrap angle into [-pi, pi]
  a = mod(a + pi, 2*pi) - pi;
endfunction
