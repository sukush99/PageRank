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


def load_website_content_and_index(content_dir, url2id, graph, maxtoload):
    directory_path = Path(content_dir)
    
    inverted_index = {}
    doc_freq = {}
    docs = {}  # {doc_id: [length, [linked_ids], {"url": ..., "filename": ...}]}
    total = 0
    for full_path in directory_path.iterdir():
        total += 1
        if (total == maxtoload): return docs, inverted_index, doc_freq
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            continue

        match = re.search(r"<!--\s*URL:\s*(https?://\S+)\s*-->", raw)
        if not match:
            continue

        original_url = match.group(1)
        doc_id = url2id.get(original_url)
        if doc_id is None:
            continue

        soup = BeautifulSoup(raw, 'html.parser')
        text = soup.get_text(separator=' ')
        tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]
        doc_len = len(tokens)

        for term in tokens:
            if term not in inverted_index:
                inverted_index[term] = {}
            if doc_id not in inverted_index[term]:
                inverted_index[term][doc_id] = 0
            inverted_index[term][doc_id] += 1

        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

        linked_ids = graph.get(doc_id, [])

        docs[doc_id] = [
            doc_len,
            linked_ids,
            {"url": original_url, "filename": str(full_path.name)}
        ]

    return docs, inverted_index, doc_freq


def compute_bm25(query, inv_index, doc_freq, docs, k1=1.5, b=0.75):
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

    # Sort by descending score
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

def interactive_search(docs, inv_index, doc_freq, base_path):
    print("Enter a search query to see top BM25 results, or enter a document ID to view the local webpage.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Query or Document ID: ").strip()
        
        if user_input.lower() == "exit":
            break

        if user_input.isdigit():
            doc_id = int(user_input)
            if doc_id in docs:
                local_filename = docs[doc_id][2]["filename"]
                full_path = os.path.join(base_path, local_filename)
                open_as_html(full_path)
            else:
                print(f"Document ID {doc_id} not found.")
        else:
            results = compute_bm25(user_input, inv_index, doc_freq, docs)
            if not results:
                print("No results found.")
                continue

            print(f"\nTop results for: '{user_input}'")
            for doc_id, score in list(results.items())[:10]:
                meta = docs[doc_id][2]
                print(f"[{doc_id}] {meta['url']} - Score: {score:.4f}")
            print()

def main():
    print("Loading ID mapping...")
    url2id = load_id_to_url_mapping("../Webb_Spam_Corpus_graph_files/url_id_mapping")

    print("Loading graph...")
    graph = load_links("../Webb_Spam_Corpus_graph_files/url_with_redirects_graph_file")

    print("Loading documents...")
    docs, inv_index, doc_freq = load_website_content_and_index(CONTENT_PATH, url2id, graph, 1000)
    print(f"Loaded {len(docs)} documents.")

    save_pickle(docs, "docs.pkl")
    save_pickle(inv_index, "inverted_index.pkl")
    save_pickle(doc_freq, "doc_freq.pkl")


    interactive_search(docs, inv_index, doc_freq, CONTENT_PATH)

if __name__ == "__main__":
    main()
