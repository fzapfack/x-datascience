SELECT DISTINCT S.sname, count(C.pid)           
FROM suppliers S, catalog C
WHERE S.sid = C.sid
GROUP BY S.sid
HAVING count(C.pid) >=2
ORDER BY S.sname;
