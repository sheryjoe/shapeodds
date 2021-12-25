function bound = getPieceWiseLinearBound(r)

  [A, b, err] = battlse(r);
  bound(1).a = [b(1)+err A(1,1) 0];
  bound(1).l = -inf;
  for i = 2:r
    bound(i).a = [b(i)+err A(i,1) 0];
    xstar = (b(i)-b(i-1))/(A(i-1,1)-A(i,1));
    bound(i-1).h = xstar;
    bound(i).l = xstar;
  end
  bound(r).h = inf;
