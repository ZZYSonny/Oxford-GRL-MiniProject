import torch
import torch.nn.functional as F
import torch_geometric.data as tgdata
import torch_geometric.utils as tgutils

def gen_edge_index(n_node, edge_prob):
    if edge_prob > 0.99:
        return torch.cartesian_prod(
            torch.arange(n_node)+n_node,
            torch.arange(n_node)
        ).T
    
    adj_float = torch.rand((n_node, n_node))
    adj_float[torch.arange(n_node), torch.arange(n_node)] = 0
    adj_bool = adj_float < edge_prob

    edge_index = tgutils.dense_to_sparse(adj_bool)[0]
    edge_index[0] += n_node
    return edge_index




def gen_dataset(n_dim, n_node, edge_prob, n_train, n_valid, n_test):
    # n: number of node in the left part of the bipartite graph
    # n_train: number of training samples
    # n_valid: number of validation samples
    torch.manual_seed(233)
    n_sample = n_train + n_valid + n_test

    # randomly generate labels for the right part of the bipartite graph
    r_labels = torch.cat([
        torch.randperm(n_node)
        for i in range(10*n_sample)
    ]).reshape(-1,n_node)
    # remove duplicates to avoid leaking
    r_labels = torch.unique(r_labels, dim=0)[0:n_train+n_valid + n_test]
    l_labels = r_labels
    
    query = torch.rand((n_sample, n_node, n_dim))
    mask = torch.cat([
        torch.ones(n_node, dtype=torch.bool),
        torch.zeros(n_node, dtype=torch.bool)
    ],dim=0)

    graph_list = [
        tgdata.Data(
            num_nodes=2*n_node,
            query=torch.vstack(
                [query[i], query[i]],
            ),
            key=torch.cat([
                torch.zeros(n_node, dtype=torch.long),
                r_labels[i]
            ],dim=0),
            y=l_labels[i],
            edge_index=gen_edge_index(n_node, edge_prob),
            mask=mask
        )
        for i in range(n_sample)
    ]
    
    train_list = graph_list[0:n_train]
    valid_list = graph_list[n_train:n_train+n_valid]
    test_list  = graph_list[n_train+n_valid:n_sample]
    return train_list, valid_list, test_list