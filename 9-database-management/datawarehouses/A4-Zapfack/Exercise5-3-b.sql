SELECT a.id as id, a.name as author, p.name as title, numpages.CASE as pages
	FROM authors as a, papers as p, paperauths as pa,
		(SELECT p2.id,
			CASE
			WHEN p2.pages ~ E'^\\d+-\\d+$' THEN
			NULLIF(SPLIT_PART(p2.pages,'-',2), '')::int - NULLIF(SPLIT_PART(p2.pages,'-',1), '')::int
			ELSE 0
			END
		FROM papers p2) as numpages,
		(SELECT a.id as id, MAX(numpages.CASE) as maxi
		FROM authors as a, papers as p, paperauths as pa, 
			(SELECT p2.id,
				CASE
				WHEN p2.pages ~ E'^\\d+-\\d+$' THEN
				NULLIF(SPLIT_PART(p2.pages,'-',2), '')::int - NULLIF(SPLIT_PART(p2.pages,'-',1), '')::int
				ELSE 0
				END
			FROM papers p2) as numpages
		WHERE a.id = pa.authid and p.id = pa.paperid and numpages.id = p.id
		GROUP BY a.id) as temp
WHERE a.id = pa.authid and p.id = pa.paperid and numpages.id = p.id and numpages.CASE = temp.maxi and temp.id = a.id
ORDER BY a.id ASC
LIMIT 10;
