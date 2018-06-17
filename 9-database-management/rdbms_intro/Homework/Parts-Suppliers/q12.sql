SELECT DISTINCT P.pname, S.sname, C.cost
FROM parts P, suppliers S, catalog C
WHERE C.pid = P.pid AND 
   C.sid = S.sid AND 
   C.cost = (SELECT min(Cbis.cost)
      FROM catalog Cbis
    WHERE Cbis.pid = P.pid)
ORDER BY P.pname;
