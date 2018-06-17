CREATE TEMP TABLE R AS
	SELECT a.id, a.name, v.year, count(p.id)
	FROM authors as a, venue as v, papers as p, paperauths as pa
	WHERE a.id = pa.authid and v.id = p.venue and p.id = pa.paperid
	GROUP BY a.id, v.year;


SELECT id, name, year, max(count) OVER w
	FROM R
	WINDOW w AS (PARTITION BY id ROWS BETWEEN 1 PRECEDING
AND 1 FOLLOWING)
	ORDER BY id ASC, year ASC
	limit 10;


