// Counts
MATCH (d:Disease) RETURN count(d) AS diseases;
MATCH (g:Gene)    RETURN count(g) AS genes;

// Shared genes between disease pairs
MATCH (d1:Disease)-[:ASSOCIATES_WITH]->(g:Gene)<-[:ASSOCIATES_WITH]-(d2:Disease)
WHERE d1 <> d2
RETURN d1.name AS disease1, d2.name AS disease2,
       collect(DISTINCT g.symbol) AS shared_genes,
       size(collect(DISTINCT g))  AS n_shared
ORDER BY n_shared DESC LIMIT 10;

// Shortest path example
MATCH p = shortestPath( (d1:Disease {name:'Asthma'})-[*..4]-(d2:Disease {name:'Breast carcinoma'}) )
RETURN p;
