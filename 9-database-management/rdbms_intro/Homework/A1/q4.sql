SELECT DISTINCT S.sname 
FROM suppliers S, 
((SELECT DISTINCT C.sid
FROM catalog C, parts P
WHERE P.color='Red' and C.cost<100 and C.pid = P.pid )
                  UNION
(SELECT DISTINCT C.sid
FROM catalog C, parts P
WHERE P.color='Green' and C.cost<100 and C.pid = P.pid  )) AS RedGreen
WHERE S.sid = RedGreen.sid
ORDER BY S.sname;

