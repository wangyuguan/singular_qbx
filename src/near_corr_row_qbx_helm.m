function [row, info] = near_corr_row_qbx_helm(xt, lam, t, D, opts, zk)

is_laplace = abs(zk) < 1e-12;

k = D.k;
nch1 = D.nch1;
nch2 = D.nch2;
lam_splits = D.lam_splits;
t_splits = D.t_splits;
alpha = D.alpha;

n_lam_near = opts.n_lam_near;
n_t_near = opts.n_t_near;
nsub = opts.nsub;
k_sub = opts.k_sub;
n_lam_near_sub = opts.n_lam_near_sub;
n_t_near_sub = opts.n_t_near_sub;
P = opts.P;
k_dya = opts.k_dya;
add_grad = opts.add_grad;
fact = 1/2;
r = .25;




N = nch1*nch2*k^2;
ncomp = 1+3*add_grad;
row = complex(zeros(ncomp, N));
info.nqbx = 0;
patch_idx = @(i, j) ((j-1)*nch1+(i-1))*k^2+(1:k^2);


% find the near panels whose contributions need to be corrected
itar = find(lam >= lam_splits, 1, 'last');
jtar = find(t >= t_splits, 1, 'last');
if isempty(itar)
    itar = 1;
end
if isempty(jtar)
    jtar = 1;
end
itar = min(max(itar, 1), nch1);
jtar = min(max(jtar, 1), nch2);

I_near = max(1, itar - n_lam_near) : min(nch1, itar + n_lam_near);
J_near = mod((jtar - n_t_near : jtar + n_t_near) - 1, nch2) + 1;

J_sort = sort(J_near);
wrap = numel(J_sort) > 1 && any(diff(J_sort) > 1);
if ~wrap
    tL = t_splits(J_sort(1));
    tR = t_splits(J_sort(end) + 1);
else
    d = diff(J_sort);
    gap_idx = find(d > 1, 1);
    low_block = J_sort(1:gap_idx);
    high_block = J_sort(gap_idx+1:end);
    tL = t_splits(high_block(1)) - 2*pi;
    tR = t_splits(low_block(end) + 1);
end

t_lift = t;
if t_lift < tL
    t_lift = t_lift + 2*pi;
end
if t_lift > tR
    t_lift = t_lift - 2*pi;
end

% remove contribution of near smooth quadrature

q_idx = repelem(1:k, k);

for ii = 1:numel(I_near)
    i = I_near(ii);
    lam_i = D.lam_nodes(:, i);
    wlam_i = D.wlam_nodes(:, i);
    for jj = 1:numel(J_near)
        j = J_near(jj);
        t_j = D.t_nodes(:, j);
        wt_j = D.wt_nodes(:, j);
        gm_j = D.gam_tpatch(:, :, j);
        jc_j = D.jac_tpatch(:, j);
        [Lam, ~] = ndgrid(lam_i, t_j);
        [Wlam, Wt] = ndgrid(wlam_i, wt_j);
        xq = gm_j(:, q_idx) .* Lam(:).';
        w_node = Lam(:) .* jc_j(q_idx.') .* Wlam(:) .* Wt(:);
        contrib = direct_block(xt, xq, w_node, zk, add_grad);
        row(:, patch_idx(i, j)) = row(:, patch_idx(i, j)) - contrib;
    end
end


% split the near panels who will be corrected

lam_sub_splits = [];
for ii = 1:numel(I_near)
    i = I_near(ii);
    lam_new = linspace(lam_splits(i), lam_splits(i+1), nsub+1);
    lam_sub_splits = [lam_sub_splits, lam_new];
end
lam_sub_splits = unique(lam_sub_splits);

t_sub_splits = [];
for jj = 1:numel(J_near)
    j = J_near(jj);
    ta = t_splits(j);
    tb = t_splits(j+1);
    mid = (ta + tb) / 2;
    if mid > tR
        ta = ta - 2*pi;
        tb = tb - 2*pi;
    end
    if mid < tL
        ta = ta + 2*pi;
        tb = tb + 2*pi;
    end
    t_sub_splits = [t_sub_splits, linspace(ta, tb, nsub+1)];
end
t_sub_splits = sort(t_sub_splits);

tol_dup = 1e-11 * max(1, abs(t_sub_splits(end)));
t_sub_splits = t_sub_splits([true, diff(t_sub_splits) > tol_dup]);

nlam_sub = numel(lam_sub_splits) - 1;
nt_sub = numel(t_sub_splits) - 1;

if nlam_sub < 1 || nt_sub < 1
    return;
end

isub = find(lam >= lam_sub_splits, 1, 'last');
jsub = find(t_lift >= t_sub_splits, 1, 'last');
if isempty(isub)
    isub = 1;
end
if isempty(jsub)
    jsub = 1;
end
isub = min(max(isub, 1), nlam_sub);
jsub = min(max(jsub, 1), nt_sub);


% reservation for qbx
I_sub_near = max(1, isub - n_lam_near_sub) : min(nlam_sub, isub + n_lam_near_sub);
J_sub_near = max(1, jsub - n_t_near_sub) : min(nt_sub, jsub + n_t_near_sub);

sub_refmask = false(nlam_sub, nt_sub);
sub_refmask(I_sub_near, J_sub_near) = true;

t_sub_valid = false(1, nt_sub);
for js = 1:nt_sub
    mid = mod((t_sub_splits(js) + t_sub_splits(js+1)) / 2, 2*pi);
    for jj2 = 1:numel(J_near)
        j2 = J_near(jj2);
        if mid >= t_splits(j2) && mid < t_splits(j2+1)
            t_sub_valid(js) = true;
            break;
        end
    end
end


% add contribution of direct quadrature from panels
% that are NOT reserved by qbx

for is = 1:nlam_sub
    a = lam_sub_splits(is);
    b = lam_sub_splits(is+1);
    [lam_s, wlam_s] = r_quad(k_sub, a, b, alpha);
    lam_s = lam_s(:);
    wlam_s = wlam_s(:);
    lam_mid = (a + b) / 2;
    i_par = find(lam_splits <= lam_mid, 1, 'last');
    i_par = min(max(i_par, 1), nch1);
    t_lam = 2*(lam_s - lam_splits(i_par))/(lam_splits(i_par+1)-lam_splits(i_par)) - 1;
    if D.is_edge(i_par) == 1
        Vr = jacobieval(t_lam, alpha, 0, k-1);
        Lr = diag((1 - lam_s).^alpha) * Vr;
    else
        Vr = legeeval(t_lam, k-1);
        Lr = Vr;
    end
    A_lam = Lr * D.v2c_lam(:, :, i_par);
    for js = 1:nt_sub
        if ~t_sub_valid(js) || sub_refmask(is, js)
            continue;
        end
        ta = t_sub_splits(js);
        tb = t_sub_splits(js+1);
        [t_s, wt_s] = lgwt(k_sub, ta, tb);
        t_s = t_s(:);
        wt_s = wt_s(:);
        t_mid = mod((ta + tb) / 2, 2*pi);
        j_par = find(t_splits <= t_mid, 1, 'last');
        j_par = min(max(j_par, 1), nch2);
        a_t = t_splits(j_par);
        b_t = t_splits(j_par+1);
        t_sh = round((mean(t_s) - (a_t + b_t)/2) / (2*pi)) * (2*pi);
        t_par = 2*(t_s - t_sh - a_t) / (b_t - a_t) - 1;
        Vt = legeeval(t_par, k-1);
        A_t = Vt*D.v2c_t(:, :, j_par);
        M_synth = kron(A_t, A_lam);
        t_s_mod = mod(t_s, 2*pi);
        gm_s = zeros(3, k_sub);
        jc_s = zeros(k_sub, 1);
        for q = 1:k_sub
            g = D.gamma(t_s_mod(q));
            dg = D.dgamma(t_s_mod(q));
            gm_s(:, q) = g;
            jc_s(q) = norm(cross(g, dg));
        end
        q_idx_s = repelem(1:k_sub, k_sub);
        [LAM_S, ~] = ndgrid(lam_s, t_s);
        [WLAM_S, WT_S] = ndgrid(wlam_s, wt_s);
        pos_s = gm_s(:, q_idx_s) .* LAM_S(:).';
        w_node_s = LAM_S(:) .* jc_s(q_idx_s.') .* WLAM_S(:) .* WT_S(:);
        contrib = direct_block(xt, pos_s, w_node_s, zk, add_grad);
        row(:, patch_idx(i_par, j_par)) = row(:, patch_idx(i_par, j_par)) + contrib*M_synth;
    end
end



% dyadic refinement for qbx reserved panels

lamL_qbx = lam_sub_splits(I_sub_near(1));
lamR_qbx = lam_sub_splits(I_sub_near(end) + 1);
tL_qbx = t_sub_splits(J_sub_near(1));
tR_qbx = t_sub_splits(J_sub_near(end) + 1);

% expansion radius = fact * (distance from the target to the panel boundary)
ns = 50;
tt_s = linspace(tL_qbx, tR_qbx, ns);
ll_s = linspace(lamL_qbx, lamR_qbx, ns);
g_edge = zeros(3, ns);
for ie = 1:ns
    g_edge(:, ie) = D.gamma(mod(tt_s(ie), 2*pi));
end
gL = D.gamma(mod(tL_qbx, 2*pi));
gR = D.gamma(mod(tR_qbx, 2*pi));
bd = [lamL_qbx*g_edge, lamR_qbx*g_edge, gL*ll_s, gR*ll_s];
delta = fact * min(vecnorm(bd - xt, 2, 1));

% halve lam and t together until the target panel diameter (max distance
% between the 4 flat corners) drops below r * delta
corners = [lamL_qbx*gL, lamL_qbx*gR, lamR_qbx*gL, lamR_qbx*gR];
diam = 0;
for a = 1:4
    diam = max(diam, max(vecnorm(corners - corners(:, a), 2, 1)));
end
nch = 1;
while diam > r*delta && nch < 30
    diam = diam/2;
    nch = nch + 1;
end
% nch=nch+2;

lamsegs = dyadic_to_point(nch, lamL_qbx, lamR_qbx, lam);
tsegs = dyadic_to_point(nch, tL_qbx, tR_qbx, t_lift);

extra_lam = lam_splits(lam_splits > lamL_qbx & lam_splits < lamR_qbx);
lamsegs = unique([lamsegs(:); extra_lam(:)]);

t_cbounds = [];
for jjc = 1:numel(J_near)
    j2 = J_near(jjc);
    ta2 = t_splits(j2);
    tb2 = t_splits(j2+1);
    m2 = (ta2 + tb2) / 2;
    if m2 > tR
        ta2 = ta2 - 2*pi;
        tb2 = tb2 - 2*pi;
    end
    if m2 < tL
        ta2 = ta2 + 2*pi;
        tb2 = tb2 + 2*pi;
    end
    t_cbounds = [t_cbounds, ta2, tb2];
end
extra_t = unique(t_cbounds);
extra_t = extra_t(extra_t > tL_qbx & extra_t < tR_qbx);
tsegs = unique([tsegs(:); extra_t(:)]);

nlam_dya = numel(lamsegs) - 1;
nt_dya = numel(tsegs) - 1;

lam_mat = zeros(k_dya, nlam_dya);
wlam_mat = zeros(k_dya, nlam_dya);
t_mat = zeros(k_dya, nt_dya);
wt_mat = zeros(k_dya, nt_dya);
for ir = 1:nlam_dya
    [rv, wrv] = r_quad(k_dya, lamsegs(ir), lamsegs(ir+1), alpha);
    lam_mat(:, ir) = rv(:);
    wlam_mat(:, ir) = wrv(:);
end
for it = 1:nt_dya
    [tv, wtv] = lgwt(k_dya, tsegs(it), tsegs(it+1));
    t_mat(:, it) = tv(:);
    wt_mat(:, it) = wtv(:);
end

lam_all = lam_mat(:);
wlam_all = wlam_mat(:);
t_all = t_mat(:);
wt_all = wt_mat(:);

n_lam_all = numel(lam_all);
n_t_all = numel(t_all);
len = n_lam_all * n_t_all;
info.nqbx = len;
q_flat = floor((0:len-1)' / n_lam_all) + 1;
p_flat = mod((0:len-1)', n_lam_all) + 1;

gm_t_all = zeros(3, n_t_all);
jc_t_all = zeros(n_t_all, 1);
for q = 1:n_t_all
    g = D.gamma(mod(t_all(q), 2*pi));
    dg = D.dgamma(mod(t_all(q), 2*pi));
    cp = cross(g, dg);
    gm_t_all(:, q) = g;
    jc_t_all(q) = norm(cp);
end

[LAM_DYA, ~] = ndgrid(lam_all, t_all);
[WLAM_DYA, WT_DYA] = ndgrid(wlam_all, wt_all);
xs = gm_t_all(:, q_flat.').*lam_all(p_flat).';
w_geo = LAM_DYA(:).*jc_t_all(q_flat).*WLAM_DYA(:).*WT_DYA(:)/(4*pi);


% contribution of each dyadic node
w_qbx = complex(zeros(len, ncomp));


g_star = D.gamma(mod(t, 2*pi));
dg_star = D.dgamma(mod(t, 2*pi));
cp_star = cross(g_star, dg_star);
nhat = cp_star/norm(cp_star);
xs_star = lam * g_star;
sgn = sign(dot(nhat, xt - xs_star));
if sgn == 0
    sgn = 1;
end
ndir = nhat * sgn;

ctr = xt + delta * ndir;
rx = norm(xt - ctr);
dx_unit = (xt - ctr)/rx;
dy = xs - ctr;
ry = vecnorm(dy, 2, 1);
cs_ang = sum(dy.*(xt - ctr), 1)./(ry*rx);

if add_grad
    [Pall, Pall_prime] = legeeval_with_deriv(cs_ang, P);
else
    Pall = legeeval(cs_ang, P);
end

if is_laplace
    n_vec = 0:P;
    radial = (rx.^n_vec)./(ry(:).^(n_vec + 1));
    w_qbx(:, 1) = w_geo.*sum(Pall.*radial, 2);
    if add_grad
        radial_d = (n_vec.*rx.^(n_vec - 1))./(ry(:).^(n_vec + 1));
        radial_sum = sum(Pall.*radial_d, 2);
        angular_sum = sum(Pall_prime.*radial, 2);
        pref = w_geo;
    end
else
    jn_rx = sph_jn(zk*rx, P);
    hn_ry = sph_h1(zk*ry, P);
    M = Pall.*hn_ry.';
    coefs = (1j*zk)*((2*(0:P) + 1).'.*jn_rx);
    w_qbx(:, 1) = w_geo.*(M*coefs);
    if add_grad
        jn_ext = sph_jn(zk*rx, P+1);
        jn_rx_prime = sph_jn_deriv(zk*rx, P, jn_ext);
        n_vec = (0:P).';
        radial_sum = M*((2*n_vec + 1).*(zk*jn_rx_prime));
        angular_sum = (Pall_prime.*hn_ry.')*((2*n_vec + 1).*jn_rx);
        pref = (1j*zk)*w_geo;
    end
end

if add_grad
    radial_term = radial_sum - (cs_ang(:)/rx).*angular_sum;
    inv_rxry = 1./(rx*ry(:));
    for d = 1:3
        w_qbx(:, 1+d) = pref.*(dx_unit(d)*radial_term + dy(d, :).'.*inv_rxry.*angular_sum);
    end
end



W2 = reshape(w_qbx, n_lam_all, n_t_all, ncomp);

A_t_all = cell(nt_dya, 1);
jpar_all = zeros(nt_dya, 1);
for it = 1:nt_dya
    t_col = t_mat(:, it);
    t_mid_d = mod((tsegs(it) + tsegs(it+1))/2, 2*pi);
    j_par = find(t_splits <= t_mid_d, 1, 'last');
    j_par = min(max(j_par, 1), nch2);
    a_t = t_splits(j_par);
    b_t = t_splits(j_par+1);
    t_sh = round((mean(t_col) - (a_t + b_t)/2)/(2*pi))*(2*pi);
    t_par_d = 2*(t_col - t_sh - a_t)/(b_t - a_t) - 1;
    A_t_all{it} = legeeval(t_par_d, k-1)*D.v2c_t(:, :, j_par);
    jpar_all(it) = j_par;
end

for ir = 1:nlam_dya
    lam_col = lam_mat(:, ir);
    lam_mid_d = (lamsegs(ir) + lamsegs(ir+1))/2;
    i_par = find(lam_splits <= lam_mid_d, 1, 'last');
    i_par = min(max(i_par, 1), nch1);
    t_lam_d = 2*(lam_col - lam_splits(i_par))/(lam_splits(i_par+1) - lam_splits(i_par)) - 1;
    if D.is_edge(i_par) == 1
        Vr_d = jacobieval(t_lam_d, alpha, 0, k-1);
        Lr_d = diag((1 - lam_col).^alpha)*Vr_d;
    else
        Vr_d = legeeval(t_lam_d, k-1);
        Lr_d = Vr_d;
    end
    A_lam_d = Lr_d*D.v2c_lam(:, :, i_par);
    lam_rows_d = (ir-1)*k_dya + (1:k_dya);
    for it = 1:nt_dya
        idx = patch_idx(i_par, jpar_all(it));
        t_cols_d = (it-1)*k_dya + (1:k_dya);
        A_t_d = A_t_all{it};
        for dc = 1:ncomp
            C = A_lam_d.'*W2(lam_rows_d, t_cols_d, dc)*A_t_d;
            row(dc, idx) = row(dc, idx) + reshape(C, 1, k^2);
        end
    end
end

end


function contrib = direct_block(xt, xq, w_node, zk, add_grad)
[Sv, Sx, Sy, Sz] = slp_grad(xt, xq, w_node, zk);
if add_grad
    contrib = [Sv.'; Sx.'; Sy.'; Sz.'];
else
    contrib = Sv.';
end
end
