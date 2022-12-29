import torch
import torch.nn.functional as F
import torch_geometric.data as tgdata

def gen_dataset(n, n_train, n_valid, n_test):
    # n: number of node in the left part of the bipartite graph
    # r_high: the highest value of the random number
    # n_train: number of training samples
    # n_valid: number of validation samples

    r_high = n
    # complete bipartite graph
    edge_index = torch.tensor([
        [r,l]
        for l in range(n)
        for r in range(n,2*n)
    ]).T

    # randomly generate labels for the right part of the bipartite graph
    r_labels = torch.cat([
        torch.randperm(n)
        for i in range(2*(n_train+n_valid+n_test))
    ]).reshape(-1,n)
    # remove duplicates to avoid leaking
    r_labels = torch.unique(r_labels, dim=0)[0:n_train+n_valid + n_test]
    l_labels = r_labels

    x_l = torch.cat([torch.eye(n), torch.zeros(n,n)], dim=1)
    x = [
        torch.cat([
            x_l,
            torch.cat([torch.eye(n), F.one_hot(r_labels[i], n)], dim=1)
        ], dim=0)
        for i in range(n_train+n_valid+n_test)
    ]

    mask = torch.cat([
        torch.ones(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool)
    ],dim=0)

    graph_list = [
        tgdata.Data(
            x=x[i].float(),
            y=l_labels[i],
            edge_index=edge_index,
            mask=mask
        )
        for i in range(n_train+n_valid+n_test)
    ]
    train_list = graph_list[0:n_train]
    valid_list = graph_list[n_train:n_train+n_valid]
    test_list  = graph_list[n_train+n_valid:n_train+n_valid+n_test]
    return train_list, valid_list, test_list