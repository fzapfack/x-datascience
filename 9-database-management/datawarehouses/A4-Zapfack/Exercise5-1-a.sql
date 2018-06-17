SELECT name, sum(publi) as spubli
FROM yearPublish 
GROUP BY name
ORDER BY spubli DESC
LIMIT 10;






