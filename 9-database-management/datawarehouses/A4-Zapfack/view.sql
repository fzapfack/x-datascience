CREATE VIEW yearPublish(id, name, year, publi) AS
SELECT a.id, a.name, v.year, count(p.id)
FROM authors as a, venue as v, papers as p, paperauths as pa
WHERE a.id = pa.authid and v.id = p.venue and p.id = pa.paperid
GROUP BY a.name, v.year;



