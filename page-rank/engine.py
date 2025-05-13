import os
import re
import math
from collections import defaultdict
from bs4 import BeautifulSoup
from pathlib import Path
import pickle
import webbrowser
import urllib.parse
import urllib.request
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import XMLParsedAsHTMLWarning
import warnings
from concurrent.futures import ProcessPoolExecutor
import mmap

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def open_as_html(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # Define path to temp HTML file in project directory
    tmp_path = os.path.join(os.getcwd(), 'preview.html')

    # Copy original content into it
    shutil.copyfile(filepath, tmp_path)

    # Open it as a file:// URL
    file_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(tmp_path))
    webbrowser.open_new_tab(file_url)

# Constants
CONTENT_PATH = "D:/Dev/WebbSpamCorpus/WebbSpamCorpus/WebbSpamCorpus"
STOPWORDS = {
    'the','is','at','which','on','and','a','an','of','or','in','to','for','with','by',
    'that','this','it','as','are','from','be','was','were','has','have','had','but','not'
}

def load_id_to_url_mapping(path):
    """
    Load ID to URL mapping.
    Returns: dict[url] = ID
    """
    url2id = {}
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                node_id, url = parts
                url2id[url] = int(node_id)
    return url2id

def load_links(path):
    """
    Load graph structure (directed edges).
    Returns: dict[doc_id] = list of linked doc_ids
    """
    graph = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src_id = int(parts[0])
            dest_ids = list(map(int, parts[1:]))
            graph[src_id].extend(dest_ids)
    return graph

def tokenize(text):
    """
    Simple tokenizer: lowercase, extract words, remove stopwords.
    """
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def load_website_content_and_index_fast(
    content_dir, url2id, graph, maxtoload, workers=4,
    report_every=10000, save_every=100000, output_prefix="savingforcrash",
    resume=0
):
    """
    Fast loader with checkpoint/resume support.
    resume: number of documents already processed and saved in checkpoint.
    """
    inverted_index = {}
    doc_freq      = {}
    docs          = {}

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    STOPWORDScache = STOPWORDS

    def process_entry(entry_path):
        try:
            raw = open(entry_path, 'r', encoding='utf-8', errors='ignore').read()
        except Exception:
            return None

        m = re.search(r"<!--\s*URL:\s*(https?://\S+)\s*-->", raw)
        if not m:
            return None

        url = m.group(1)
        doc_id = url2id.get(url)
        if doc_id is None:
            return None
        try:
            soup = BeautifulSoup(raw, 'html.parser')
            text = soup.get_text(" ")
        except:
            text = ""

        tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDScache]
        if not tokens:
            return None

        local_index    = {}
        for t in tokens:
            local_index[t] = local_index.get(t, 0) + 1
        local_doc_freq = {t: 1 for t in set(tokens)}

        linked_ids = graph.get(doc_id, [])
        meta = {"url": url, "filename": os.path.basename(entry_path), "pagerank_score": 0}
        return (doc_id, len(tokens), local_index, local_doc_freq, linked_ids, meta)

    # 1) Collect file paths
    file_paths = []
    for entry in os.scandir(content_dir):
        if entry.is_file():
            file_paths.append(entry.path)
            if len(file_paths) >= maxtoload:
                break

    print(f"Found {len(file_paths)} files for processing. Resuming at {resume}.")
    # 2) Optionally load resume checkpoint
    if resume > 0:
        docs = load_pickle(f"{output_prefix}docs_{resume}.pkl")
        inverted_index = load_pickle(f"{output_prefix}inverted_index_{resume}.pkl")
        doc_freq = load_pickle(f"{output_prefix}doc_freq_{resume}.pkl")
        print(f"Loaded checkpoint at {resume} documents.")

    file_paths_to_process = file_paths[resume:]

    # 3) Process remaining in parallel
    with ThreadPoolExecutor(max_workers=workers) as exe:
        for local_idx, result in enumerate(exe.map(process_entry, file_paths_to_process), 1):
            global_i = resume + local_idx
            if global_i % report_every == 0:
                print(f"Processed {global_i} documents...")
            if global_i % save_every == 0:
                print(f"Saving intermediate structures at {global_i} documents...")
                save_pickle(docs, f"{output_prefix}docs_{global_i}.pkl")
                save_pickle(inverted_index, f"{output_prefix}inverted_index_{global_i}.pkl")
                save_pickle(doc_freq, f"{output_prefix}doc_freq_{global_i}.pkl")
            if result is None:
                continue
            doc_id, doc_len, loc_idx, loc_df, linked, meta = result
            # Merge inverted index
            for term, cnt in loc_idx.items():
                if term not in inverted_index:
                    inverted_index[term] = {}
                inverted_index[term][doc_id] = inverted_index[term].get(doc_id, 0) + cnt
            # Merge doc freq
            for term in loc_df:
                doc_freq[term] = doc_freq.get(term, 0) + 1
            docs[doc_id] = [doc_len, linked, meta]

    # 4) Compute PageRank and attach
    pagerank_scores = compute_pagerank(graph)
    for doc_id, score in pagerank_scores.items():
        if doc_id in docs:
            docs[doc_id][2]["pagerank_score"] = score

    return docs, inverted_index, doc_freq



def compute_bm25_pagerank_search(query, inv_index, doc_freq, docs, k1=1.5, b=0.75, pagerank_weight=0.2):
    """
    Compute BM25 scores for a query.
    Returns: dict[doc_id] = score
    """
    scores = {}
    query_terms = tokenize(query)
    N = len(docs)
    avg_dl = sum(length for length, _, _ in docs.values()) / N

    for term in query_terms:
        if term not in inv_index:
            continue

        df = doc_freq.get(term, 0)
        if df == 0:
            continue

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        for doc_id, tf in inv_index[term].items():
            doc_len = docs[doc_id][0]
            denom = tf + k1 * (1 - b + b * doc_len / avg_dl)
            score = idf * ((tf * (k1 + 1)) / denom)
            scores[doc_id] = scores.get(doc_id, 0) + score

    max_bm25 = max(scores.values())
    # normalise and combine bm25 and pagerank
    max_pr = max(docs[doc_id][2]['pagerank_score'] for doc_id in scores if 'pagerank_score' in docs[doc_id][2])
    for doc_id in scores:
        norm_bm25 = scores[doc_id] / max_bm25 if max_bm25 > 0 else 0
        norm_pr = docs[doc_id][2].get('pagerank_score', 0) / max_pr if max_pr > 0 else 0
        scores[doc_id] = (1 - pagerank_weight) * norm_bm25 + pagerank_weight * norm_pr

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

def build_reverse_graph(graph):
    reverse_graph = {node: [] for node in graph}
    for src, outlinks in graph.items():
        for dst in outlinks:
            if dst in reverse_graph:
                reverse_graph[dst].append(src)
    return reverse_graph

def compute_pagerank(graph, damping=0.85, max_iter=20, tol=1.0e-6):
    N = len(graph)
    pr = {node: 1.0 / N for node in graph}
    print("building reverse graph")
    reverse_graph = build_reverse_graph(graph)

    for _ in range(max_iter):
        new_pr = {}
        for node in graph:
            rank_sum = 0.0
            for incoming in reverse_graph.get(node, []):
                out_deg = len(graph[incoming])
                if out_deg > 0:
                    rank_sum += pr[incoming] / out_deg
            new_pr[node] = (1 - damping) / N + damping * rank_sum

        delta = sum(abs(new_pr[n] - pr[n]) for n in pr)
        pr = new_pr
        if delta < tol:
            break

    return pr

def interactive_search(docs, inv_index, doc_freq, base_path):
    print("Enter a search query to see top BM25 results, or enter a document ID to view the local webpage.")
    print("Type 'pagerankweight n' to set the pagerank weight in the search, where n is a float from 0 to 1")
    print("Type 'exit' to quit.\n")
    pagerank_weight=0.2
    while True:
        user_input = input("Query or Document ID: ").strip()
        
        if user_input.lower() == "exit":
            break
        
        if user_input.lower().startswith("pagerankweight "):
            try:
                new_weight = float(user_input.split()[1])
                if 0 <= new_weight <= 1:
                    pagerank_weight = new_weight
                    print(f"PageRank weight set to {pagerank_weight}, higher means pagerank has more impact")
                else:
                    print("Weight must be between 0 and 1.")
            except (IndexError, ValueError):
                print("Invalid format. Use: weight 0.3")
            continue

        if user_input.isdigit():
            doc_id = int(user_input)
            if doc_id in docs:
                local_filename = docs[doc_id][2]["filename"]
                full_path = os.path.join(base_path, local_filename)
                open_as_html(full_path)
            else:
                print(f"Document ID {doc_id} not found.")
        else:
            results = compute_bm25_pagerank_search(user_input, inv_index, doc_freq, docs,pagerank_weight)
            if not results:
                print("No results found.")
                continue

            print(f"\nTop results for: '{user_input}'")
            for doc_id, score in list(results.items())[:10]:
                meta = docs[doc_id][2]
                print(f"[{doc_id}] {meta['url']} - BM25+pagerank weighted at: {pagerank_weight} Score: {score:.4f}")
            print()

def main():
    parser = argparse.ArgumentParser(description="WebbSpamCorpus BM25 Indexer with resume")
    parser.add_argument('--load-pickle', action='store_true', help='Load data from final pickle files instead of rebuilding.')
    parser.add_argument('--resume', type=int, default=0, help='Resume processing from a checkpoint at given document count')
    args = parser.parse_args()

    if args.load_pickle:
        print("Loading search acceleration structures from final pickle files...")
        docs = load_pickle("docs.pkl")
        inv_index = load_pickle("inverted_index.pkl")
        doc_freq = load_pickle("doc_freq.pkl")
    else:
        print("Loading ID mapping...")
        url2id = load_id_to_url_mapping("../Webb_Spam_Corpus_graph_files/url_id_mapping")

        print("Loading graph...")
        graph = load_links("../Webb_Spam_Corpus_graph_files/url_with_redirects_graph_file")

        print("Loading documents... This may take a while.")
        docs, inv_index, doc_freq = load_website_content_and_index_fast(
            CONTENT_PATH, url2id, graph, 800000,
            workers=4, report_every=10000, save_every=100000,
            output_prefix="savingforcrash", resume=args.resume
        )
        print(f"Loaded {len(docs)} documents.")


        print("Saving final structures...")
        save_pickle(docs, "docs.pkl")
        save_pickle(inv_index, "inverted_index.pkl")
        save_pickle(doc_freq, "doc_freq.pkl")

    interactive_search(docs, inv_index, doc_freq, CONTENT_PATH)

if __name__ == "__main__":
    main()
