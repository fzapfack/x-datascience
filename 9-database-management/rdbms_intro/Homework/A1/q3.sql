SELECT DISTINCT S.sname
	FROM suppliers S, catalog C, parts P
	WHERE P.color='Red' and C.cost<100 and C.pid = P.pid and S.sid = C.sid 
	ORDER BY S.sname;
