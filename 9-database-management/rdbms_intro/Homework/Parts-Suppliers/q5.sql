SELECT DISTINCT S.sname 
FROM suppliers S
WHERE S.sid IN 
(SELECT DISTINCT C.sid
FROM catalog C, parts P
WHERE P.color='Red' and C.cost<100 and C.pid = P.pid )
     and S.sid IN 
(SELECT DISTINCT C.sid
FROM catalog C, parts P
WHERE P.color='Green' and C.cost<100 and C.pid = P.pid )
ORDER BY S.sname;
