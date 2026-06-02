clear; clc;
addpath('../../software/chebfun')
addpath('src/')

zk = 1.0;
lam_inner = 0.90;
k = 8;
nch1 = 2;
nch2 = 24;

gamma = @(t) [cos(t); sin(t); 0];
dgamma = @(t) [-sin(t); cos(t); 0];
t_splits = linspace(0, 2*pi, nch2+1);

theta = 2*pi*rand;
x = [0.8*cos(theta); 0.8*sin(theta); 0]


Gkern = @(d) exp(1i*zk*d)./(4*pi*d);





alpha = -.5;
D = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, alpha);

r = D.lam_nodes_all;
rho = (1 - r).^alpha;
d = vecnorm(D.src_xyz - x, 2, 1).';
S_quad = sum(D.src_w.*Gkern(d).*rho);

integ = @(R, T) Gkern(sqrt((R.*cos(T) - x(1)).^2 + (R.*sin(T) - x(2)).^2 + x(3).^2)).*(1 - R).^alpha.*R;
S_ref = integral2(integ, lam_inner, 1, 0, 2*pi, 'RelTol', 1e-12, 'AbsTol', 1e-14, 'Method', 'iterated');

rel_err = abs(S_quad - S_ref)/abs(S_ref);
fprintf('  rel err     = %.2e\n', rel_err);

