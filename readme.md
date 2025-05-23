### Hybrid Search Engine Ranking

This project illustrates the implementation of a search engine that combines a content-based ranking algorithm (BM25) and link-based authority scoring (PageRank).
Overview

For this project, we've leveraged the WebbSpamCorpus dataset to demonstrate how a search engine based on these two algorithms efficiently indexes web page content and constructs the underlying link relation between the pages.
#### BM25

The BM25 algorithm elevates web page importance based on:

    Term frequency
    Inverse document frequency
    Document length normalization

#### PageRank

The PageRank algorithm iteratively calculates the global page importance.
Combined Scoring

The **normalized scores** of these two algorithms, including a user-adjusted weight factor, can produce relevant results to a user.
