function Q = precompute_helm_qbx_corr(targ_xyz, lam_vec, t_vec, D, opts, zk)

is_grad = opts.is_grad;
N = numel(D.src_w);
nt = size(targ_xyz, 2);

ii_cell = cell(nt, 1);
jj_cell = cell(nt, 1);
vx_cell = cell(nt, 1);
vy_cell = cell(nt, 1);
vz_cell = cell(nt, 1);

fprintf('Precomputing correction from boundary layer\n');
% parfor t = 1:nt
for t=1:nt
    xt = targ_xyz(:, t);
    row_t = near_corr_row_qbx_helm(xt, lam_vec(t), t_vec(t), D, opts, zk)/(4*pi);
    nz = find(any(row_t ~= 0, 1));
    ii_cell{t} = repmat(t, numel(nz), 1);
    jj_cell{t} = nz(:);
    if is_grad
        vx_cell{t} = row_t(1, nz).';
        vy_cell{t} = row_t(2, nz).';
        vz_cell{t} = row_t(3, nz).';
    else
        vx_cell{t} = row_t(1, nz).';
    end
end

ii = vertcat(ii_cell{:});
jj = vertcat(jj_cell{:});
vx = vertcat(vx_cell{:});

if opts.is_grad
    vy = vertcat(vy_cell{:});
    vz = vertcat(vz_cell{:});
    Q.Sx = sparse(ii, jj, vx, nt, N);
    Q.Sy = sparse(ii, jj, vy, nt, N);
    Q.Sz = sparse(ii, jj, vz, nt, N);
else
    Q.S = sparse(ii, jj, vx, nt, N);
end
end
