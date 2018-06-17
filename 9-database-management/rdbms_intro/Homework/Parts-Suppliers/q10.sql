SELECT DISTINCT P.pname, max(C.cost), avg(C.cost)
FROM catalog C, parts P
WHERE C.pid = P.pid 
GROUP BY P.pid
ORDER BY P.pname;
