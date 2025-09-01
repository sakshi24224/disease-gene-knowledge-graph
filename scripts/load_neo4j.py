import argparse
import pandas as pd
from py2neo import Graph
from tqdm import tqdm

C_MERGE = """
MERGE (d:Disease {name: $disease})
MERGE (g:Gene   {symbol: $gene})
MERGE (d)-[r:ASSOCIATES_WITH]->(g)
SET   r.score = $score
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="bolt://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="neo4j1234")  # same as Docker login
    p.add_argument("--csv", default="data/disgenet_sample.csv")
    args = p.parse_args()

    graph = Graph(args.uri, auth=(args.user, args.password))

    # run each constraint as a single-line statement
    graph.run("CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
    graph.run("CREATE CONSTRAINT gene_symbol IF NOT EXISTS FOR (g:Gene) REQUIRE g.symbol IS UNIQUE")

    df = pd.read_csv(args.csv)
    df["score"] = pd.to_numeric(df.get("score", 0.5), errors="coerce").fillna(0.5)

    tx = graph.begin()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tx.run(C_MERGE, disease=row["disease_name"].strip(), gene=row["gene_symbol"].strip(), score=float(row["score"]))
    graph.commit(tx)
    print(f"Loaded rows: {len(df)}")

if __name__ == "__main__":
    main()
