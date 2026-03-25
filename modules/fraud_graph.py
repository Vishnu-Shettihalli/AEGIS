import networkx as nx
import plotly.graph_objects as go

def build_fraud_graph(df):

    G = nx.Graph()

    fraud_df = df[df["fraud"]==1].head(200)

    for _,row in fraud_df.iterrows():

        customer = row["customer"]
        merchant = row["merchant"]

        G.add_node(customer,type="customer")
        G.add_node(merchant,type="merchant")

        G.add_edge(customer,merchant)

    pos = nx.spring_layout(G)

    edge_x=[]
    edge_y=[]

    for edge in G.edges():

        x0,y0 = pos[edge[0]]
        x1,y1 = pos[edge[1]]

        edge_x.extend([x0,x1,None])
        edge_y.extend([y0,y1,None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1,color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    node_x=[]
    node_y=[]
    colors=[]

    for node in G.nodes():

        x,y = pos[node]

        node_x.append(x)
        node_y.append(y)

        if G.nodes[node]["type"]=="customer":
            colors.append("#00FFC6")
        else:
            colors.append("#FF4D4D")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=12,
            color=colors
        )
    )

    fig = go.Figure(data=[edge_trace,node_trace])

    fig.update_layout(
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        showlegend=False
    )

    return fig