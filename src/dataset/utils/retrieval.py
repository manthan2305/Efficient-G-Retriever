import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
from torch.nn.functional import cosine_similarity

def retrieval_via_attention(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, threshold_node=None, threshold_edge=None):
    
    # If no textual descriptions, return original graph and its description.
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + \
               textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc
    
    # Compute node attention scores using cosine similarity between q_emb and node features.
    node_scores = cosine_similarity(q_emb, graph.x)  # shape: (num_nodes,)
    
    # Select nodes based on threshold or topk selection.
    if threshold_node is None:
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(node_scores, topk, largest=True)
    else:
        topk_n_indices = (node_scores >= threshold_node).nonzero(as_tuple=False).view(-1)
        if topk_n_indices.numel() == 0:
            topk = min(topk, graph.num_nodes)
            _, topk_n_indices = torch.topk(node_scores, topk, largest=True)
    
    # Compute edge attention scores similarly.
    edge_scores = cosine_similarity(q_emb, graph.edge_attr)  # shape: (num_edges,)
    
    if threshold_edge is None:
        unique_edge_scores = edge_scores.unique()
        topk_e = min(topk_e, unique_edge_scores.size(0))
        topk_e_values, _ = torch.topk(unique_edge_scores, topk_e, largest=True)
        # Select edges that have scores above the smallest topk_e value.
        selected_edge_mask = edge_scores >= topk_e_values[-1]
        topk_e_indices = selected_edge_mask.nonzero(as_tuple=False).view(-1)
    else:
        topk_e_indices = (edge_scores >= threshold_edge).nonzero(as_tuple=False).view(-1)
        if topk_e_indices.numel() == 0:
            unique_edge_scores = edge_scores.unique()
            topk_e = min(topk_e, unique_edge_scores.size(0))
            topk_e_values, _ = torch.topk(unique_edge_scores, topk_e, largest=True)
            selected_edge_mask = edge_scores >= topk_e_values[-1]
            topk_e_indices = selected_edge_mask.nonzero(as_tuple=False).view(-1)
    
    # Aggregate nodes that are incident to the selected high-attention edges.
    row, col = graph.edge_index
    selected_edge_nodes = torch.cat([row[topk_e_indices], col[topk_e_indices]]).unique()
    
    # Take the union of the top nodes and the nodes incident to high-attention edges.
    selected_nodes = torch.unique(torch.cat([topk_n_indices, selected_edge_nodes]))
    
    # Build a set for quick lookup.
    selected_nodes_set = set(selected_nodes.tolist())
    
    # Filter edges: keep those where both endpoints are in the selected set.
    selected_edge_indices = []
    for i in range(graph.edge_index.shape[1]):
        u = graph.edge_index[0, i].item()
        v = graph.edge_index[1, i].item()
        if u in selected_nodes_set and v in selected_nodes_set:
            selected_edge_indices.append(i)
    selected_edge_indices = torch.tensor(selected_edge_indices, dtype=torch.long, device=graph.x.device)
    
    # Create textual description using selected nodes and edges.
    n = textual_nodes.iloc[selected_nodes.cpu().numpy()]
    e = textual_edges.iloc[selected_edge_indices.cpu().numpy()]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    
    # Remap node indices to create a contiguous subgraph.
    mapping = {int(n): i for i, n in enumerate(selected_nodes.cpu().numpy())}
    edge_index_sub = graph.edge_index[:, selected_edge_indices]
    src = [mapping[int(i)] for i in edge_index_sub[0].tolist()]
    dst = [mapping[int(i)] for i in edge_index_sub[1].tolist()]
    edge_index_sub = torch.LongTensor([src, dst]).to(graph.x.device)
    
    # Construct the new subgraph Data object.
    x_sub = graph.x[selected_nodes]
    edge_attr_sub = graph.edge_attr[selected_edge_indices]
    data = Data(x=x_sub, edge_index=edge_index_sub, edge_attr=edge_attr_sub, num_nodes=selected_nodes.size(0))
    
    return data, desc

def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc
