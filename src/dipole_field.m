function E = dipole_field(x, x0, p, zk)
df = x-x0;
r2 = sum(df.^2, 1);
r = sqrt(r2);
eikr = exp(1i*zk*r);
yp = sum(df.*p, 1);
coef1 = (zk^2*r2+1i*zk*r-1)./(4*pi*r2.*r).*eikr;
coef2 = (3-3*1i*zk*r-zk^2*r2)./(4*pi*r2.^2.*r).*eikr;
E = coef1.*p + coef2.*yp.*df;
end
