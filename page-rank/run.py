from config import config
import networkx as nx

def parse_edge_list(path):
    G = nx.DiGraph() # directional graph

    # Manually parse the edge list
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()  # Split by spaces
            if len(parts) > 1:
                node = int(parts[0])  # First value is the node ID
                neighbors = map(int, parts[1:])  # Remaining values are outgoing links
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
    
    return G

def parse_id_mapping(path):
    url_mapping = {}

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)  # Prevents splitting URLs with spaces
            if len(parts) == 2:
                node_id, url = parts
                url_mapping[int(node_id)] = url

    return url_mapping

def compute_pagerank(G):
    # Compute PageRank scores
    pagerank_scores = nx.pagerank(G, alpha=0.85)  # Alpha is the damping factor

    #sort list by score
    sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_pagerank

def main():
    Link = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
    print(f"Link: {Link}")
    print(f"Link from config: {config.link}")

    

    # Load the graph from an edge list file
    G = parse_edge_list(config.graph_path)
    Id_to_url = parse_id_mapping(config.idmap_path)

    # Print basic info about the graph
    print(G)
    sorted_by_pagerank = compute_pagerank(G)

    #highest 10 scoring page ids
    for node, score in sorted_by_pagerank[:10]:
        print(f"Pageid: {node}, page url: {Id_to_url[node]}, Score: {score}")

    


if __name__ == "__main__":
    main()
