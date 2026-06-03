clear; clc;
addpath('../chebfun')
addpath('src/')

rng(1)

zk = 1.0;
lam_inner = 0.9;
k = 8;
nch1 = 2;
nch2 = 24;

gamma = @(t) [cos(t); sin(t); 0];
dgamma = @(t) [-sin(t); cos(t); 0];
t_splits = linspace(0, 2*pi, nch2+1);


opts = [];
opts.n_lam_near = 1;
opts.n_t_near = 1;
opts.nsub = 4;
opts.k_sub = 16;
opts.n_lam_near_sub = 1;
opts.n_t_near_sub = 1;
opts.P = 20;
opts.k_dya = k;

alpha = -0.5;
D = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, alpha);


opts.add_grad = false;
opts.ncores = 1;
tar = D.tar_xyz;
lam = D.tar_lam_nodes_all;
t = D.tar_t_nodes_all;


tic 
[Q, info] = precompute_helm_qbx_corr(tar, lam, t, D, opts, zk);
t_single = toc;
nqbx = info.nqbx;
S = Q.S;

opts.ncores = 16; 
% burn in 
[Q_par, ~] = precompute_helm_qbx_corr(tar, lam, t, D, opts, zk);


tic 
[Q_par, ~] = precompute_helm_qbx_corr(tar, lam, t, D, opts, zk);
S_par = Q_par.S;
t_par = toc;

save('time_info.mat', 'nqbx', 'S', 'S_par', 't_single', 't_par')