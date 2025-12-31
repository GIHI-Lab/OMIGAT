from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx
import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def load_network(file_path):
    """
    Load network from file.
    :param file_path: Full pathname of the network file
    :return: net (class: pandas.DataFrame): Edges in the network, nodes (class: pandas.DataFrame): The nodes in the network
    """
    net = pd.read_table(filepath_or_buffer=file_path, header=None,
                        index_col=None, names=['source', 'target'], sep='\t')
    nodes = pd.concat([net['source'], net['target']], ignore_index=True)
    nodes = pd.DataFrame(nodes, columns=['nodes']).drop_duplicates()
    nodes.reset_index(drop=True, inplace=True)
    return net, nodes

def create_network(genes,edges,direction=False):
    """
    Create a networkx graph object by genes and edges.
    :param genes (class: pandas.DataFrame): List of genes
    :param edges (class: pandas.DataFrame): List of edges
    :param direction (bool, optional): Whether the graph object is directed or undirected
    :return: A networkx graph object
    """
    gene_list=genes.iloc[:,0].values.tolist()
    n=len(gene_list)
    if direction:
        G=nx.DiGraph()
    else :
        G=nx.Graph()

    for i in np.arange(0,n):
        G.add_node(gene_list[i])

    for _,row in edges.iterrows():
        G.add_edge(row['source'],row['target'],weight=1)
    return G

def sparse_dense_graph(graph_adj, topN=30):
    """
    Sparse the dense graph by retaining the topN edges for each node
    :param graph_adj: Adjacency matrix of dense graph
    :param topN (int, optional): Number of edges to retain for each node
    :return (class: numpy.array): Adjacency matrix of sparse graph
    """
    net_mtx = np.zeros((graph_adj.shape[0],graph_adj.shape[1]))
    indices = np.argsort(-graph_adj)[:,:topN]
    for i in np.arange(0, indices.shape[0]):
        for j in np.arange(0, indices.shape[1]):
            net_mtx[i, indices[i,j]] = graph_adj[i,indices[i,j]]
    return net_mtx

def convert_adj_to_edgeset(adj, node_df):
    """
    Extract edges from adjacency matrix of graph.
    :param adj: Adjacency matrix of graph
    :param node_df (class: pandas.DataFrame): node names
    :return (class: pandas.DataFrame): Edges in the graph
    """
    sour_lst = []
    targ_lst = []
    for i in np.arange(0, node_df.shape[0]):
        row = adj[i,:]
        targ = node_df.loc[row.nonzero()[0], :]['nodes'].tolist()
        sour = [node_df.iloc[i, 0] for j in range(0, len(targ))]
        targ_lst = targ_lst + targ
        sour_lst = sour_lst + sour
    return pd.DataFrame({'source':sour_lst, 'target':targ_lst})

def create_knn_graph(features, k=30, metric='cosine'):
    """
    Create KNN graph based on node features
    :param features: Node features matrix
    :param k: Number of nearest neighbors
    :param metric: Distance metric for KNN
    :return: edge_index in torch_geometric format
    """
    # Initialize KNN model
    knn = NearestNeighbors(n_neighbors=k+1, metric=metric)  # k+1 because it includes self as neighbor
    knn.fit(features)
    
    # Get K nearest neighbors
    distances, indices = knn.kneighbors()
    
    # Create edge list (excluding self-loops)
    rows = np.repeat(np.arange(indices.shape[0]), k)
    cols = indices[:, 1:].flatten()  # exclude self-loops by removing first column
    
    # Create edge weights based on distances
    weights = 1 - distances[:, 1:].flatten()  # convert distance to similarity
    
    # Create bidirectional edges
    edge_index = torch.tensor(np.vstack([
        np.concatenate([rows, cols]),
        np.concatenate([cols, rows])
    ]), dtype=torch.long)
    
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index

def generate_graph(args, dataset):
    """
    Generate graph structure based on node features using KNN
    :param args: Arguments received from command line
    :param dataset: Dataset dictionary containing features and other information
    :return: edge_index in torch_geometric format
    """
    features = dataset['feature']
    k = args.knn_k if hasattr(args, 'knn_k') else 30
    
    # Generate KNN graph
    edge_index = create_knn_graph(features, k=k)
    print(f'KNN graph with k={k} is generated successfully...')
    
    return edge_index




