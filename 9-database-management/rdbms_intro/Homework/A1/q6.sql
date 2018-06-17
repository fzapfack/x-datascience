SELECT DISTINCT C.pid
FROM catalog C, catalog CBis
WHERE C.pid = CBis.pid AND C.sid <> CBis.sid
ORDER BY C.pid;

