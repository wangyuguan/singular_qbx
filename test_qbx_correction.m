clear; clc;
addpath('../chebfun')
addpath('src/')

rng(1)

zk = 1.0;
lam_inner = 0.8999;
k = 8;
nch1 = 2;
nch2 = 24;

gamma = @(t) [cos(t); sin(t); 0];
dgamma = @(t) [-sin(t); cos(t); 0];
t_splits = linspace(0, 2*pi, nch2+1);
angfun = @(th) 1 + 0.4*cos(th) + 0.3*sin(2*th);

lam_t = 0.978;
t_t = 2*pi*rand;
x = [lam_t*cos(t_t); lam_t*sin(t_t); 0];

opts = [];
opts.n_lam_near = 1;
opts.n_t_near = 1;
opts.nsub = 4;
opts.k_sub = 16;
opts.n_lam_near_sub = 1;
opts.n_t_near_sub = 1;
opts.P = 30;
opts.nch_lam_dya = 12;
opts.nch_t_dya = 16;
opts.k_dya = 16;

alpha = -0.5;
D = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, alpha);
r = D.lam_nodes_all;
sigma = (1 - r).^alpha.*angfun(D.t_nodes_all);

Sref = @(p) integral2(@(rr, tt) slp_int(rr, tt, p, zk, alpha, angfun), lam_inner, 1, 0, 2*pi, 'RelTol', 1e-6, 'AbsTol', 1e-6);
Sxref = @(p) integral2(@(rr, tt) sx_int(rr, tt, p, zk, alpha, angfun), lam_inner, 1, 0, 2*pi, 'RelTol', 1e-6, 'AbsTol', 1e-6);
Syref = @(p) integral2(@(rr, tt) sy_int(rr, tt, p, zk, alpha, angfun), lam_inner, 1, 0, 2*pi, 'RelTol', 1e-6, 'AbsTol', 1e-6);

% hedgehog reference for S, Sx, Sy
M = 5;
h0 = 0.001;
nrm = [0; 0; 1];
Sv = zeros(M, 1);
Sxv = zeros(M, 1);
Syv = zeros(M, 1);
for i = 1:M
    p = x + i*h0*nrm;
    Sv(i) = Sref(p);
    Sxv(i) = Sxref(p);
    Syv(i) = Syref(p);
end
s = (1:M).';
V = s.^(0:M-1);
cS = V\Sv;
cX = V\Sxv;
cY = V\Syv;
S_ref = cS(1);
Sx_ref = cX(1);
Sy_ref = cY(1);

dvec = vecnorm(D.src_xyz - x, 2, 1).';
df = x - D.src_xyz;
wgt = D.src_w.*sigma;
G = exp(1i*zk*dvec)./(4*pi*dvec);
coefs = (1i*zk*dvec - 1).*exp(1i*zk*dvec)./(4*pi*dvec.^3);

opts.add_grad = true;
Q = precompute_helm_qbx_corr(x, lam_t, t_t, D, opts, zk);
S_qbx = sum(wgt.*G) + Q.S*sigma;
Sx_qbx = sum(wgt.*coefs.*df(1, :).') + Q.Sx*sigma;
Sy_qbx = sum(wgt.*coefs.*df(2, :).') + Q.Sy*sigma;

rel_S = abs(S_qbx - S_ref)/abs(S_ref)
rel_Sx = abs(Sx_qbx - Sx_ref)/abs(Sx_ref)
rel_Sy = abs(Sy_qbx - Sy_ref)/abs(Sy_ref)

function v = slp_int(rr, tt, p, zk, alpha, angfun)
dfx = p(1) - rr.*cos(tt);
dfy = p(2) - rr.*sin(tt);
r = sqrt(dfx.^2 + dfy.^2 + p(3).^2);
v = exp(1i*zk*r)./(4*pi*r).*(1 - rr).^alpha.*angfun(tt).*rr;
end

function v = sx_int(rr, tt, p, zk, alpha, angfun)
dfx = p(1) - rr.*cos(tt);
dfy = p(2) - rr.*sin(tt);
r = sqrt(dfx.^2 + dfy.^2 + p(3).^2);
v = (1i*zk*r - 1).*exp(1i*zk*r)./(4*pi*r.^3).*dfx.*(1 - rr).^alpha.*angfun(tt).*rr;
end

function v = sy_int(rr, tt, p, zk, alpha, angfun)
dfx = p(1) - rr.*cos(tt);
dfy = p(2) - rr.*sin(tt);
r = sqrt(dfx.^2 + dfy.^2 + p(3).^2);
v = (1i*zk*r - 1).*exp(1i*zk*r)./(4*pi*r.^3).*dfy.*(1 - rr).^alpha.*angfun(tt).*rr;
end
