import numpy as np
import pandas as pd
import networkx as nx

edges = pd.DataFrame()
edges['sources'] = [1,1,1,2,2,3,3,4,4,5,5,5]
edges['targets'] = [2,4,5,3,1,2,5,1,5,1,3,4]
edges['weights'] = [1,1,1,1,1,1,1,1,1,1,1,1]
# sources are the begin nodes
# targets are the end nodes
# weights are the weight on the edge

def show_edges():
    # show our define variable
    print(edges)
    # define a graph
    G = nx.from_pandas_edgelist(edges, source='sources',target='targets',edge_attr='weights')
    # print graph
    print(G)
    # degree
    print('degree(度): ',nx.degree(G))
    # connnected component
    print('connected components(连通分量): ',list(nx.connected_components(G)))
    # graph diameter
    print('graph diameter(图直径): ',nx.diameter(G))
    # degree contrality
    print('degree_contrality(度中心性)',nx.degree_centrality(G))
    # eigenvector centrality
    print('eigenvector centrality(特征向量中心性): ',nx.eigenvector_centrality(G))
    # betweenness centrality
    print('betweeness centrality(中介中心性): ',nx.betweenness_centrality(G))
    # closeness centrality
    print('closeness centrality(连接中心性): ',nx.closeness_centrality(G))
    # pagerank
    print('pagerank(图排序): ',nx.pagerank(G))
    # HITS
    print('HITS(图排序): ',nx.hits(G))
    print("HITS中前面的是hubs值，后面的是authorities值。")




if __name__ == '__main__':
    print("Welcome to test.py")
    show_edges()
