from experiments.graphs.graph_generator import GRAPH_TYPES
from experiments.graphs.graph_serializer import serialize_graph, count_tokens

for s in [100, 200, 500]:
    g = GRAPH_TYPES["erdos_renyi"](n_nodes=s, seed=42)
    text = serialize_graph(g, "adjacency_list")
    tokens = count_tokens(text)
    print(f"n={s}: nodes={g.number_of_nodes()}, edges={g.number_of_edges()}, tokens={tokens}, chars={len(text)}")
