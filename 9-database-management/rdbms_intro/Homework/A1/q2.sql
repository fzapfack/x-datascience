SELECT  DISTINCT  P.pname, P.color
	FROM suppliers S, catalog C, parts P
	WHERE S.sname = 'Perfunctory Parts' and S.sid = C.sid and C.pid = P.pid 
	ORDER BY P.pname, P.color; 
