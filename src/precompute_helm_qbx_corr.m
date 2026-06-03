function [Q, info] = precompute_helm_qbx_corr(targ_xyz, lam_vec, t_vec, D, opts, zk)


add_grad = opts.add_grad;
ncores = opts.ncores;
N = numel(D.src_w);
nt = size(targ_xyz, 2);


ii_cell = cell(nt, 1);
jj_cell = cell(nt, 1);
vS_cell = cell(nt, 1);
vx_cell = cell(nt, 1);
vy_cell = cell(nt, 1);
vz_cell = cell(nt, 1);
nqbx = zeros(nt, 1);

fprintf('Precomputing correction from boundary layer\n');
if ncores == 1
    for t = 1:nt
        xt = targ_xyz(:, t);
        [row_t, info_t] = near_corr_row_qbx_helm(xt, lam_vec(t), t_vec(t), D, opts, zk);
        nqbx(t) = info_t.nqbx;
        nz = find(any(row_t ~= 0, 1));
        ii_cell{t} = repmat(t, numel(nz), 1);
        jj_cell{t} = nz(:);
        vS_cell{t} = row_t(1, nz).';
        if add_grad
            vx_cell{t} = row_t(2, nz).';
            vy_cell{t} = row_t(3, nz).';
            vz_cell{t} = row_t(4, nz).';
        end
    end
else
    parfor (t = 1:nt, ncores)
        xt = targ_xyz(:, t);
        [row_t, info_t] = near_corr_row_qbx_helm(xt, lam_vec(t), t_vec(t), D, opts, zk);
        nqbx(t) = info_t.nqbx;
        nz = find(any(row_t ~= 0, 1));
        ii_cell{t} = repmat(t, numel(nz), 1);
        jj_cell{t} = nz(:);
        vS_cell{t} = row_t(1, nz).';
        if add_grad
            vx_cell{t} = row_t(2, nz).';
            vy_cell{t} = row_t(3, nz).';
            vz_cell{t} = row_t(4, nz).';
        end
    end
end

ii = vertcat(ii_cell{:});
jj = vertcat(jj_cell{:});
Q.S = sparse(ii, jj, vertcat(vS_cell{:}), nt, N);
if add_grad
    Q.Sx = sparse(ii, jj, vertcat(vx_cell{:}), nt, N);
    Q.Sy = sparse(ii, jj, vertcat(vy_cell{:}), nt, N);
    Q.Sz = sparse(ii, jj, vertcat(vz_cell{:}), nt, N);
end


info.nqbx = nqbx;
end
