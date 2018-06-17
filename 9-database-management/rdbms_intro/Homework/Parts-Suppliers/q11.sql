SELECT DISTINCT  P.pname, S.sname, C.cost
FROM parts P, suppliers S, catalog C
WHERE C.pid = P.pid AND C.sid = S.sid AND 
   C.cost IN (SELECT min(CBis.cost)
       FROM catalog CBis WHERE CBis.pid = P.pid)
ORDER BY P.pname ;
