SELECT a.id as id, a.name as author, p.name as title, numpages.CASE as pages, AVG(avgpages.CASE)
FROM authors as a, papers as p, paperauths as pa, 
	(SELECT p2.id,
		CASE
		WHEN p2.pages ~ E'^\\d+-\\d+$' THEN
		NULLIF(SPLIT_PART(p2.pages,'-',2), '')::int - NULLIF(SPLIT_PART(p2.pages,'-',1), '')::int
		ELSE 0
		END
	FROM papers p2) as numpages, 
	(SELECT p3.id,
		CASE
		WHEN p3.pages ~ E'^\\d+-\\d+$' THEN
		NULLIF(SPLIT_PART(p3.pages,'-',2), '')::int - NULLIF(SPLIT_PART(p3.pages,'-',1), '')::int
		ELSE 0
		END
	FROM papers p3) as avgpages
WHERE a.id = pa.authid and p.id = pa.paperid and numpages.id = p.id and avgpages.id in (select p2.id as id 
FROM authors as a2, papers as p2, paperauths as pa2
WHERE a2.id = a.id and a2.id = pa2.authid and p2.id = pa2.paperid)/* Makes the excecution very long */
GROUP BY a.id, p.name, numpages.CASE
ORDER BY a.id ASC
LIMIT 10;


/*==>>> Take too long to run */


