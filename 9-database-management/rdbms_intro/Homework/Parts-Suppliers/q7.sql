SELECT DISTINCT S.sname
FROM suppliers S
WHERE NOT EXISTS ( SELECT P.pid
      FROM parts P
      WHERE P.color = 'Red'  AND NOT EXISTS
        ( SELECT C.pid
          FROM catalog C
           WHERE C.pid = P.pid AND C.sid = S.sid))
ORDER BY S.sname ;
