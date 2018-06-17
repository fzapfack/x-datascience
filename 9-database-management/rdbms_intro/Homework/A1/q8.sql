SELECT DISTINCT S.sname
FROM suppliers S, catalog C
WHERE S.sid = C.sid AND NOT EXISTS 
(SELECT *
  FROM catalog CBis
  WHERE CBis.sid = S.sid AND CBis.cost >= 100)
ORDER BY S.sname;

