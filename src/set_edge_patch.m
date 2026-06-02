function D = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, alpha)

lam_splits = lam_inner + (1 - lam_inner) * (0:nch1) / nch1;

lam_nodes = zeros(k, nch1);
wlam_nodes = zeros(k, nch1);
for i = 1:nch1
    [lam, wlam] = r_quad(k, lam_splits(i), lam_splits(i+1), alpha);
    lam_nodes(:, i) = lam(:);
    wlam_nodes(:, i) = wlam(:);
end

t_nodes = zeros(k, nch2);
wt_nodes = zeros(k, nch2);
for j = 1:nch2
    [t, wt] = lgwt(k, t_splits(j), t_splits(j+1));
    t_nodes(:, j) = t(:);
    wt_nodes(:, j) = wt(:);
end

is_edge = zeros(nch1, 1);
for i = 1:nch1
    is_edge(i) = abs(lam_splits(i+1) - 1) < 1e-12;
end

D_lege = diag((2*(0:k-1)' + 1) / 2);

v2c_lam = zeros(k, k, nch1);
for i = 1:nch1
    a = lam_splits(i);
    b = lam_splits(i+1);
    t_par = 2*(lam_nodes(:, i) - a) / (b - a) - 1;
    if is_edge(i) == 1
        w_jac = wlam_nodes(:, i) .* (1 - lam_nodes(:, i)).^alpha;
        v2c = jacobieval(t_par, alpha, 0, k-1);
        norm_cons = (v2c.^2).' * w_jac;
        v2c_lam(:, :, i) = diag(1 ./ norm_cons) * (v2c.' .* wlam_nodes(:, i).');
    else
        w_std = wlam_nodes(:, i) * (2 / (b - a));
        v2c = legeeval(t_par, k-1);
        v2c_lam(:, :, i) = D_lege * (v2c.' .* w_std.');
    end
end

v2c_t = zeros(k, k, nch2);
for j = 1:nch2
    a = t_splits(j);
    b = t_splits(j+1);
    t_par = 2*(t_nodes(:, j) - a) / (b - a) - 1;
    w_std = wt_nodes(:, j)*2 / (b - a);
    v2c = legeeval(t_par, k-1);
    v2c_t(:, :, j) = D_lege*(v2c.'.*w_std.');
end

gam_tpatch = zeros(3, k, nch2);
jac_tpatch = zeros(k, nch2);
nhat_tpatch = zeros(3, k, nch2);
for j = 1:nch2
    tj = t_nodes(:, j);
    gm = zeros(3, k);
    dg = zeros(3, k);
    for q = 1:k
        gm(:, q) = gamma(tj(q));
        dg(:, q) = dgamma(tj(q));
    end
    cp = cross(gm, dg);
    jac = vecnorm(cp).';
    gam_tpatch(:, :, j) = gm;
    jac_tpatch(:, j) = jac;
    nhat_tpatch(:, :, j) = cp ./ jac.';
end

N = nch1 * nch2 * k^2;
q_idx = repelem(1:k, k).';

lam_nodes_all = zeros(N, 1);
wlam_nodes_all = zeros(N, 1);
t_nodes_all = zeros(N, 1);
wt_nodes_all = zeros(N, 1);
src_xyz = zeros(3, N);
src_w = zeros(N, 1);
src_n = zeros(3, N);

for j = 1:nch2
    gm_j = gam_tpatch(:, :, j);
    jc_j = jac_tpatch(:, j);
    nh_j = nhat_tpatch(:, :, j);
    tj = t_nodes(:, j);
    wtj = wt_nodes(:, j);
    for i = 1:nch1
        lam_i = lam_nodes(:, i);
        wlam_i = wlam_nodes(:, i);
        [lam, t] = ndgrid(lam_i, tj);
        [wlam, wt] = ndgrid(wlam_i, wtj);
        i0 = ((j-1)*nch1 + (i-1))*k^2 + 1;
        ind = i0 : i0+k^2-1;
        lam_nodes_all(ind) = lam(:);
        t_nodes_all(ind) = t(:);
        wlam_nodes_all(ind) = wlam(:);
        wt_nodes_all(ind) = wt(:);
        src_xyz(:, ind) = gm_j(:, q_idx.') .* lam(:).';
        src_w(ind) = lam(:) .* jc_j(q_idx) .* wlam(:) .* wt(:);
        src_n(:, ind) = nh_j(:, q_idx.');
    end
end

tar_xyz = zeros(3, N);
tar_w = zeros(N, 1);
tar_n = zeros(3, N);
tar_lam_nodes_all = zeros(N, 1);
tar_t_nodes_all = zeros(N, 1);
tar_wlam_nodes_all = zeros(N, 1);
tar_wt_nodes_all = zeros(N, 1);

for j = 1:nch2
    gm_j = gam_tpatch(:, :, j);
    jc_j = jac_tpatch(:, j);
    nh_j = nhat_tpatch(:, :, j);
    tj = t_nodes(:, j);
    wtj = wt_nodes(:, j);
    for i = 1:nch1
        if is_edge(i) == 1
            [lam_i, wlam_i] = lgwt(k, lam_splits(i), lam_splits(i+1));
        else
            lam_i = lam_nodes(:, i);
            wlam_i = wlam_nodes(:, i);
        end
        lam_i = lam_i(:);
        wlam_i = wlam_i(:);
        [lam, t] = ndgrid(lam_i, tj);
        [wlam, wt] = ndgrid(wlam_i, wtj);
        i0 = ((j-1)*nch1 + (i-1))*k^2 + 1;
        ind = i0 : i0+k^2-1;
        tar_lam_nodes_all(ind) = lam(:);
        tar_t_nodes_all(ind) = t(:);
        tar_wlam_nodes_all(ind) = wlam(:);
        tar_wt_nodes_all(ind) = wt(:);
        tar_xyz(:, ind) = gm_j(:, q_idx.') .* lam(:).';
        tar_w(ind) = lam(:) .* jc_j(q_idx) .* wlam(:) .* wt(:);
        tar_n(:, ind) = nh_j(:, q_idx.');
    end
end

nfine = max(200, 10*k*nch2);
t_fine = linspace(0, 2*pi, nfine+1)';
t_fine = t_fine(1:end-1);
gam_fine = zeros(3, nfine);
for q = 1:nfine
    gam_fine(:, q) = gamma(t_fine(q));
end
gam_fine_norm2 = sum(gam_fine.^2, 1);

D.k = k;
D.nch1 = nch1;
D.nch2 = nch2;
D.alpha = alpha;
D.lam_inner = lam_inner;
D.lam_splits = lam_splits;
D.t_splits = t_splits;
D.gamma = gamma;
D.dgamma = dgamma;
D.is_edge = is_edge;
D.lam_nodes = lam_nodes;
D.wlam_nodes = wlam_nodes;
D.t_nodes = t_nodes;
D.wt_nodes = wt_nodes;
D.v2c_lam = v2c_lam;
D.v2c_t = v2c_t;
D.gam_tpatch = gam_tpatch;
D.jac_tpatch = jac_tpatch;
D.nhat_tpatch = nhat_tpatch;
D.lam_nodes_all = lam_nodes_all;
D.wlam_nodes_all = wlam_nodes_all;
D.t_nodes_all = t_nodes_all;
D.wt_nodes_all = wt_nodes_all;
D.src_xyz = src_xyz;
D.src_w = src_w;
D.src_n = src_n;
D.tar_xyz = tar_xyz;
D.tar_w = tar_w;
D.tar_n = tar_n;
D.tar_lam_nodes_all = tar_lam_nodes_all;
D.tar_t_nodes_all = tar_t_nodes_all;
D.tar_wlam_nodes_all = tar_wlam_nodes_all;
D.tar_wt_nodes_all = tar_wt_nodes_all;
D.t_fine = t_fine;
D.gam_fine = gam_fine;
D.gam_fine_norm2 = gam_fine_norm2;

end
