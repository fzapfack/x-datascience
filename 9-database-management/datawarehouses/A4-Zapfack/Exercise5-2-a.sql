SELECT avg(x.c)
FROM (SELECT count(name) as c
FROM yearPublish 
WHERE publi=1
GROUP BY year) as x;







