import argparse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="data/embeddings_node2vec.csv")
    p.add_argument("--disease", default="Asthma")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", default="", help="Optional path to save results as CSV")
    args = p.parse_args()

    # rows indexed by node ids like "D:Asthma", "G:IL13"
    emb = pd.read_csv(args.emb, index_col=0)

    disease_rows = emb[emb.index.str.startswith("D:")]
    if disease_rows.empty:
        print("No disease embeddings found. Run embeddings first.")
        return

    q_key = f"D:{args.disease}"
    if q_key not in disease_rows.index:
        print(f"Disease not in embeddings: {args.disease}")
        return

    # cosine similarity to all other diseases
    q_vec = disease_rows.loc[[q_key]].values  # 1 x d
    sims = cosine_similarity(q_vec, disease_rows.values).flatten()

    out = pd.DataFrame({"key": disease_rows.index, "cosine": sims})
    out = out[out["key"] != q_key].sort_values("cosine", ascending=False).head(args.topk)
    out["disease"] = out["key"].str.replace("^D:", "", regex=True)
    out = out[["disease", "cosine"]]

    print(f"Nearest diseases to '{args.disease}' by embedding:")
    print(out.to_string(index=False, formatters={"cosine": "{:.4f}".format}))

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
