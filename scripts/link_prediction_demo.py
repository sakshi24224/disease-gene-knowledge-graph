import argparse
import pandas as pd
from py2neo import Graph
from sklearn.metrics.pairwise import cosine_similarity

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="bolt://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="neo4j1234")
    p.add_argument("--emb", default="data/embeddings_node2vec.csv")
    p.add_argument("--disease", default="Asthma")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", default="")
    args = p.parse_args()

    emb = pd.read_csv(args.emb, index_col=0)
    D = emb[emb.index.str.startswith("D:")]
    G = emb[emb.index.str.startswith("G:")]
    dq = f"D:{args.disease}"
    if dq not in D.index:
        print(f"Disease not found in embeddings: {args.disease}")
        return

    graph = Graph(args.uri, auth=(args.user, args.password))
    known_df = pd.DataFrame(graph.run("""
        MATCH (:Disease {name:$d})-[:ASSOCIATES_WITH]->(g:Gene)
        RETURN g.symbol AS gene
    """, d=args.disease).data())
    known = set(known_df["gene"]) if not known_df.empty else set()

    q_vec = D.loc[[dq]].values
    sims = cosine_similarity(q_vec, G.values).flatten()

    out = pd.DataFrame({"gene_key": G.index, "cosine": sims})
    out["gene"] = out["gene_key"].str.replace("^G:", "", regex=True)
    out = out[~out["gene"].isin(known)].sort_values("cosine", ascending=False).head(args.topk)
    print(f"\nCandidate genes for '{args.disease}' (excluding known links):")
    print(out[["gene","cosine"]].to_string(index=False, formatters={"cosine": "{:.4f}".format}))

    if args.out:
        out[["gene","cosine"]].to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
