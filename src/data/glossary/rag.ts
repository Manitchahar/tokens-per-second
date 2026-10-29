import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  rag: {
    term: 'RAG (Retrieval-Augmented Generation)',
    def: 'An architecture that retrieves relevant documents from an external corpus and injects them into the prompt context window before the LLM generates a response, giving the model access to private or recent facts without retraining.',
    lesson: 'rag/why-rag',
  },
  'parametric-memory': {
    term: 'Parametric memory',
    def: 'Knowledge stored directly inside the floating-point weights of a neural network. Diffuse, lossy, subject to a fixed cutoff date, and expensive to update via backpropagation.',
    lesson: 'rag/why-rag',
  },
  'non-parametric-memory': {
    term: 'Non-parametric memory',
    def: 'Knowledge stored outside the model in an external searchable index (such as a vector database or inverted index). Exact, auditable, permission-aware, and updatable in O(1) time without training.',
    lesson: 'rag/why-rag',
  },
  'knowledge-cutoff': {
    term: 'Knowledge cutoff',
    def: 'The point in time at which a model’s training dataset was collected. Facts, events, or software updates occurring after this timestamp cannot be recalled from model weights.',
    lesson: 'rag/why-rag',
  },
  'bi-encoder': {
    term: 'Bi-encoder (dual-tower)',
    def: 'A model architecture that encodes queries and documents into dense vectors independently. Enables O(1) approximate nearest neighbor vector search, but lacks cross-attention between query and document tokens.',
    lesson: 'rag/embeddings-search',
  },
  'cross-encoder': {
    term: 'Cross-encoder (reranker)',
    def: 'A model architecture that concatenates query and document tokens into a single sequence and runs full bidirectional self-attention across all token pairs. Highly accurate at scoring relevance, but computationally expensive (O(L^2)).',
    lesson: 'rag/chunking-reranking',
  },
  'cosine-similarity': {
    term: 'Cosine similarity',
    def: 'A metric measuring the cosine of the angle theta between two vectors: (u · v) / (||u|| * ||v||). On unit-normalized vectors, it simplifies directly to the dot product u · v.',
    lesson: 'rag/embeddings-search',
  },
  'l2-normalization': {
    term: 'L2 normalization',
    def: 'Scaling a vector by dividing each element by its Euclidean length (norm), projecting the vector onto the surface of a unit hypersphere where its length equals 1.',
    lesson: 'rag/embeddings-search',
  },
  mac: {
    term: 'MAC (Multiply-Accumulate)',
    def: 'A hardware arithmetic instruction that performs one multiplication and one addition in a single cycle (2 FLOPs). A dot product of dimension d requires d MACs.',
    lesson: 'rag/embeddings-search',
  },
  bm25: {
    term: 'BM25 (Best Matching 25)',
    def: 'The industry-standard sparse lexical ranking algorithm. Scores documents by combining inverse document frequency with saturating term frequency (governed by k1) and document length normalization (governed by b).',
    lesson: 'rag/bm25-hybrid',
  },
  'inverted-index': {
    term: 'Inverted index',
    def: 'A data structure that maps every distinct term in a corpus to a posting list of document IDs containing that term, enabling sub-millisecond keyword lookups decoupled from total corpus size.',
    lesson: 'rag/bm25-hybrid',
  },
  'posting-list': {
    term: 'Posting list',
    def: 'The list of document IDs (and term occurrences) stored under a specific keyword in an inverted index.',
    lesson: 'rag/bm25-hybrid',
  },
  'term-frequency': {
    term: 'Term frequency (TF)',
    def: 'How many times a specific keyword appears within a single document passage.',
    lesson: 'rag/bm25-hybrid',
  },
  idf: {
    term: 'IDF (Inverse document frequency)',
    def: 'A measure of how rare a term is across the entire corpus. Ubiquitous words like "the" receive an IDF near zero, while rare technical keywords receive high IDF scores.',
    lesson: 'rag/bm25-hybrid',
  },
  rrf: {
    term: 'RRF (Reciprocal Rank Fusion)',
    def: 'A rank-based fusion algorithm that combines rankings from multiple retrieval systems by summing 1 / (k + rank). Eliminates the need to normalize incompatible raw score scales.',
    lesson: 'rag/bm25-hybrid',
  },
  'hybrid-retrieval': {
    term: 'Hybrid retrieval',
    def: 'A retrieval pipeline that queries both a sparse lexical index (BM25) and a dense semantic vector index concurrently, fusing their candidate lists to capture both exact keywords and conceptual synonyms.',
    lesson: 'rag/bm25-hybrid',
  },
  ann: {
    term: 'ANN (Approximate nearest neighbours)',
    def: 'A family of algorithms and data structures that trade a small amount of recall (e.g. 1–2%) to search high-dimensional vector spaces 50x to 1,000x faster than brute-force linear scans.',
    lesson: 'rag/ann-indexes',
  },
  ivf: {
    term: 'IVF (Inverted file index)',
    def: 'An ANN indexing technique that partitions vector space into Voronoi cells using k-means clustering. At query time, only vectors inside the top nprobe nearest centroid cells are evaluated.',
    lesson: 'rag/ann-indexes',
  },
  nprobe: {
    term: 'nprobe',
    def: 'The tuning parameter in IVF indexes specifying how many neighboring Voronoi centroid cells to inspect at query time. Controls the trade-off between search recall and query latency.',
    lesson: 'rag/ann-indexes',
  },
  hnsw: {
    term: 'HNSW (Hierarchical Navigable Small World)',
    def: 'A graph-based ANN indexing structure that organizes vectors into a hierarchy of proximity skip-graphs. Searches in O(log N) time via greedy routing along long-range highway edges.',
    lesson: 'rag/ann-indexes',
  },
  'product-quantization': {
    term: 'Product quantization (PQ)',
    def: 'A vector compression technique that decomposes high-dimensional vectors into m sub-vectors and quantizes each to a 1-byte codebook centroid index, achieving up to 32x–64x memory reduction.',
    lesson: 'rag/ann-indexes',
  },
  adc: {
    term: 'ADC (Asymmetric distance computation)',
    def: 'Distance calculation between an unquantized query vector and product-quantized document vectors using precomputed centroid lookup tables rather than floating-point math.',
    lesson: 'rag/ann-indexes',
  },
  faiss: {
    term: 'FAISS',
    def: 'Facebook AI Similarity Search: an optimized C++ and Python library for dense vector clustering, indexing, and GPU/CPU nearest-neighbor search.',
    lesson: 'rag/ann-indexes',
  },
  chunking: {
    term: 'Chunking',
    def: 'The process of partitioning continuous long-form documents into discrete, semantically coherent text segments suitable for embedding and retrieval.',
    lesson: 'rag/chunking-reranking',
  },
  'chunk-overlap': {
    term: 'Chunk overlap',
    def: 'A sliding-window buffer of shared tokens between consecutive chunks that prevents sentences spanning chunk boundaries from being cut in half.',
    lesson: 'rag/chunking-reranking',
  },
  'contextual-retrieval': {
    term: 'Contextual retrieval',
    def: 'An ingestion technique introduced by Anthropic that prepends 50–100 tokens of LLM-generated document-level summary to each chunk, eliminating orphan facts and reducing retrieval failures by up to 49%.',
    lesson: 'rag/chunking-reranking',
  },
  'query-rewriting': {
    term: 'Query rewriting',
    def: 'Reformulating an ambiguous or conversational multi-turn user prompt into an explicit, self-contained search query packed with relevant keywords before querying a retrieval index.',
    lesson: 'rag/chunking-reranking',
  },
  'precision-at-k': {
    term: 'Precision@k',
    def: 'The fraction of the top-k retrieved documents that are relevant to the query: (Retrieved_k ∩ Relevant) / k.',
    lesson: 'rag/evaluating-retrieval',
  },
  'recall-at-k': {
    term: 'Recall@k',
    def: 'The fraction of total ground-truth relevant documents in the corpus captured within the top-k retrieved list: (Retrieved_k ∩ Relevant) / Total_Relevant.',
    lesson: 'rag/evaluating-retrieval',
  },
  mrr: {
    term: 'MRR (Mean Reciprocal Rank)',
    def: 'The average of reciprocal ranks (1 / rank_1) of the first relevant document across an evaluation query set.',
    lesson: 'rag/evaluating-retrieval',
  },
  dcg: {
    term: 'DCG (Discounted Cumulative Gain)',
    def: 'A ranking metric that sums graded document relevance scores, applying a logarithmic position discount: sum((2^rel - 1) / log2(i + 1)).',
    lesson: 'rag/evaluating-retrieval',
  },
  ndcg: {
    term: 'NDCG (Normalized Discounted Cumulative Gain)',
    def: 'DCG divided by the Ideal DCG (IDCG) of a perfectly ordered ranking, bounding the score between 0.0 and 1.0.',
    lesson: 'rag/evaluating-retrieval',
  },
  'rag-triad': {
    term: 'The RAG Triad',
    def: 'A three-part production evaluation framework measuring Context Relevance (retriever), Groundedness/Faithfulness (generator), and Answer Relevance (generator).',
    lesson: 'rag/evaluating-retrieval',
  },
  faithfulness: {
    term: 'Faithfulness (groundedness)',
    def: 'The proportion of claims in an AI-generated response that are strictly entailed and supported by the retrieved context, guarding against hallucinations.',
    lesson: 'rag/evaluating-retrieval',
  },
};
