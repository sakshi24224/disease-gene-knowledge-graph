import argparse
import pandas as pd
from py2neo import Graph
import networkx as nx
from node2vec import Node2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="bolt://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="neo4j1234")
    args = p.parse_args()

    graph = Graph(args.uri, auth=(args.user, args.password))

    rows = graph.run(
        "MATCH (d:Disease)-[r:ASSOCIATES_WITH]->(g:Gene) "
        "RETURN d.name AS disease, g.symbol AS gene, r.score AS score"
    ).data()
    edges = pd.DataFrame(rows)
    if edges.empty:
        print("No edges found. Load data first.")
        return

    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(f"D:{row['disease']}", f"G:{row['gene']}", weight=float(row['score']))

    n2v = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, workers=2, quiet=True)
    model = n2v.fit(window=10, min_count=1, batch_words=128)

    nodes = list(G.nodes())
    vectors = [model.wv[n] for n in nodes]
    emb = pd.DataFrame(vectors, index=nodes)
    emb.to_csv("data/embeddings_node2vec.csv")
    print("Saved embeddings to data/embeddings_node2vec.csv")

    xy = PCA(n_components=2).fit_transform(emb.values)
    plt.figure(figsize=(7,6))
    plt.scatter(xy[:,0], xy[:,1], s=25, alpha=0.8)
    plt.title("Node2Vec embeddings (PCA 2D)")
    plt.tight_layout()
    plt.savefig("data/embeddings_pca.png", dpi=150)
    print("Saved plot to data/embeddings_pca.png")

if __name__ == "__main__":
    main()
