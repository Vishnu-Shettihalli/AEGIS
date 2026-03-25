import networkx as nx

def network_risk(df):

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(row["customer"], row["merchant"])

    density = nx.density(G)

    return min(density * 10, 1)