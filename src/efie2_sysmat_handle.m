function A = efie2_sysmat_handle(I, J, target_xyz, ...
    src_xyz_J, src_w_J, src_xyz_rho, src_w_rho, ...
    i2i_S, i2i_gx, i2i_gy, i2b_S, i2b_gx, i2b_gy, ...
    b2i_S_J, b2i_gx_J, b2i_gy_J, b2b_S_J, b2b_gx_J, b2b_gy_J, ...
    b2i_S_rho, b2i_gx_rho, b2i_gy_rho, b2b_S_rho, b2b_gx_rho, b2b_gy_rho, ...
    zk, N, ni)

% Returns the submatrix A(I, J) of the 3N x 3N system. The unknown is
% as x = [Jx; Jy; rho]
%
% Representation
%       E_scat = i*k*S[J] + grad S[rho]
%
% Integral equation
%       i*k*S[Jx] + dx S[rho]                     = -Einc_x
%       i*k*S[Jy] + dy S[rho]                     = -Einc_y
%       i*k*(dx S[Jx] + dy S[Jy]) - zk^2*S[rho]   = 0
%
% System matrix on [Jx; Jy; rho]
%       [ i*k*S      0         dx S     ]
%       [ 0         i*k*S      dy S     ]
%       [ i*k*dx S  i*k*dy S  -zk^2*S   ]
%
% The near corrections are passed as separate inner/band blocks (no big M_*):
%   i2i_*  inner source -> inner target (shared by J and rho)
%   i2b_*  inner source -> band target  (shared by J and rho)
%   b2i_*_J/_rho  band source -> inner target
%   b2b_*_J/_rho  band source -> band target
% The column of the assembled M for source sj is [inner-tgt block; band-tgt
% block](:, c), rebuilt here on the fly.

A = complex(zeros(numel(I), numel(J)));

% split each global index into block (1: Jx, 2: Jy, 3: rho)
i_blk = ceil(I(:)/N);
i_loc = mod(I(:) - 1, N) + 1;
j_blk = ceil(J(:)/N);
j_loc = mod(J(:) - 1, N) + 1;

is_x = (i_blk == 1);
is_y = (i_blk == 2);
is_d = (i_blk == 3);

for q = 1:numel(J)
    sb = j_blk(q);
    sj = j_loc(q);

    if sb == 3
        src = src_xyz_rho(:, sj);
        w = src_w_rho(sj);
    else
        src = src_xyz_J(:, sj);
        w = src_w_J(sj);
    end

    % rebuild the near-correction column from the inner/band blocks
    if sj <= ni
        c = sj;
        colS = [i2i_S(:, c);  i2b_S(:, c)];
        colGx = [i2i_gx(:, c); i2b_gx(:, c)];
        colGy = [i2i_gy(:, c); i2b_gy(:, c)];
    elseif sb == 3
        c = sj - ni;
        colS = [b2i_S_rho(:, c);  b2b_S_rho(:, c)];
        colGx = [b2i_gx_rho(:, c); b2b_gx_rho(:, c)];
        colGy = [b2i_gy_rho(:, c); b2b_gy_rho(:, c)];
    else
        c = sj - ni;
        colS = [b2i_S_J(:, c);  b2b_S_J(:, c)];
        colGx = [b2i_gx_J(:, c); b2b_gx_J(:, c)];
        colGy = [b2i_gy_J(:, c); b2b_gy_J(:, c)];
    end

    % single layer and its gradient from this source to every requested target
    [S, Sx, Sy] = slp_grad(target_xyz(:, i_loc), src, w, zk);
    S = S + full(colS(i_loc));
    Sx = Sx + full(colGx(i_loc));
    Sy = Sy + full(colGy(i_loc));

    col = complex(zeros(numel(I), 1));
    if sb == 1
        % Jx:  i*k*S -> E_x row,  i*k*dx S -> div row
        col(is_x) = 1i*zk*S(is_x);
        col(is_d) = 1i*zk*Sx(is_d);
    elseif sb == 2
        % Jy:  i*k*S -> E_y row,  i*k*dy S -> div row
        col(is_y) = 1i*zk*S(is_y);
        col(is_d) = 1i*zk*Sy(is_d);
    else
        % rho:  dx S -> E_x,  dy S -> E_y,  -zk^2*S -> div
        col(is_x) = Sx(is_x);
        col(is_y) = Sy(is_y);
        col(is_d) = -zk^2*S(is_d);
    end
    A(:, q) = col;
end
end
