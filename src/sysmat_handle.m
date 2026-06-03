function A = sysmat_handle(i, j, target_xyz, ...
    src_xyz_J, src_w_J, M_S_J, M_gx_J, M_gy_J, ...
    src_xyz_rho, src_w_rho, M_S_rho, M_gx_rho, M_gy_rho, zk, N)
% sysmat_handler  Maxwell EFIE system-matrix entry handle 
%
% Returns the submatrix A(i, j) of the 3N x 3N system. The unknown is ordered
% as x = [J_u; J_v; rho]
%
% Representation
%       E_scat = i*k*S[J] + grad S[rho]
%
% Integral equation
%       row 1:  i*k*S[J_u] + dx S[rho]                     = -E_inc_x
%       row 2:  i*k*S[J_v] + dy S[rho]                     = -E_inc_y
%       row 3:  i*k*(dx S[J_u] + dy S[J_v]) - zk^2*S[rho]  = 0
%
% System matrix on [J_u; J_v; rho]  (S = single layer, dx/dy = its gradient)
%       [ i*k*S      0         dx S     ]
%       [ 0         i*k*S      dy S     ]
%       [ i*k*dx S  i*k*dy S  -zk^2*S   ]

A = complex(zeros(numel(i), numel(j)));

% assign block index (1 = J_u, 2 = J_v, 3 = rho) and local index
i_blk = ceil(i(:)/N);
i_loc = mod(i(:) - 1, N) + 1;
j_blk = ceil(j(:)/N);
j_loc = mod(j(:) - 1, N) + 1;

for q = 1:numel(j)
    sb = j_blk(q);      
    sj = j_loc(q);      

    if sb == 3
        src = src_xyz_rho(:, sj);
        w = src_w_rho(sj);
        MS = M_S_rho;
        Mx = M_gx_rho;
        My = M_gy_rho;
    else
        src = src_xyz_J(:, sj);
        w = src_w_J(sj);
        MS = M_S_J;
        Mx = M_gx_J;
        My = M_gy_J;
    end

    for tb = 1:3
        rows = find(i_blk == tb);
        if isempty(rows)
            continue;
        end
        ti = i_loc(rows);

        [S, Sx, Sy] = slp_grad(target_xyz(:, ti), src, w, zk);
        S = S + full(MS(ti, sj));       
        Sx = Sx + full(Mx(ti, sj));
        Sy = Sy + full(My(ti, sj));

        A(rows, q) = efie_block(tb, sb, zk, S, Sx, Sy);
    end
end
end


function v = efie_block(tb, sb, zk, S, Sx, Sy)
% (tb, sb) entry of the EFIE block operator
%       [ i*k*S      0         dx S     ]
%       [ 0         i*k*S      dy S     ]
%       [ i*k*dx S  i*k*dy S  -zk^2*S   ]
z = zeros(size(S));
row1 = {1i*zk*S,  z,        Sx     };
row2 = {z,        1i*zk*S,  Sy     };
row3 = {1i*zk*Sx, 1i*zk*Sy, -zk^2*S};
op = {row1, row2, row3};
v = op{tb}{sb};
end
