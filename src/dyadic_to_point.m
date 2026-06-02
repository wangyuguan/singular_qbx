function segs = dyadic_to_point(nch, a, b, t)
t = min(max(t,a),b);
if t == a
    segs = generate_dyadic_segments(nch,a,b,true);
    return
end
if t == b
    segs = generate_dyadic_segments(nch,a,b,false);
    return
end
segsL = generate_dyadic_segments(nch,a,t,false);
segsR = generate_dyadic_segments(nch,t,b,true);
segs = unique([segsL(:); segsR(:)]);
segs = sort(segs);
tol = 100*eps(max(1,abs(t)));
segs(abs(segs - t) < tol) = [];
end
