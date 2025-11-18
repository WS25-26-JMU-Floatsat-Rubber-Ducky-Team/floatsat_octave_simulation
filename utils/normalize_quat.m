function qn = normalize_quat(q)
  n = norm(q);
  if n == 0
    qn = [1 0 0 0];   % fallback unit quaternion
  else
    qn = q / n;
  end
endfunction
