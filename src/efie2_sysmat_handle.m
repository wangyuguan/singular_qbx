function A = efie2_sysmat_handle(I, J, target_xyz, ...
    src_xyz_J, src_w_J, M_S_J, M_gx_J, M_gy_J, ...
    src_xyz_rho, src_w_rho, M_S_rho, M_gx_rho, M_gy_rho, zk, N)

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

A = complex(zeros(numel(I), numel(J)));

% split each global index into block (1: Jx, 2: Jy, 3: rho) 
i_blk = ceil(I(:)/N);
i_loc = mod(I(:) - 1, N) + 1;
j_blk = ceil(J(:)/N);
j_loc = mod(J(:) - 1, N) + 1;

% which requested rows live in the E_x, E_y, and div blocks
is_x = (i_blk == 1);
is_y = (i_blk == 2);
is_d = (i_blk == 3);

for q = 1:numel(J)
    % source block
    sb = j_blk(q);     
    % source local index
    sj = j_loc(q);     

    % pick the band the source column lives on, with its near correction
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

    % single layer and its gradient from this source to every requested target
    [S, Sx, Sy] = slp_grad(target_xyz(:, i_loc), src, w, zk);
    S = S + full(MS(i_loc, sj));
    Sx = Sx + full(Mx(i_loc, sj));
    Sy = Sy + full(My(i_loc, sj));


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


