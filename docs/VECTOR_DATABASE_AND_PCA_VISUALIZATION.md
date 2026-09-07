# Vector Database Engineering & 3D PCA Visualization

## 1. Vector Store Architecture

The vector storage layer is powered by **ChromaDB**, managed through [`backend/modules/vector_db.py`](file:///home/vicky/Projects/Local_MultiAgentic_RAG_System/backend/modules/vector_db.py).

### Chunk Identification & Deduplication Schema
Each chunk is assigned a deterministic hierarchical identifier:
$$\text{chunk\_id} = \text{source\_filename} : \text{page\_number} : \text{chunk\_index}$$

*Example*: `Constitution Of India 2022.pdf:14:2` (Page 14, Chunk index 2).

This prevents duplicate documents or repeated pages from bloating the vector store on subsequent ingestion cycles:
```python
existing_items = self.db.get(include=[])
existing_ids = set(existing_items["ids"])

# Filter out previously ingested chunks
new_chunks = [c for c in chunks_with_ids if c.metadata["chunk_id"] not in existing_ids]
if new_chunks:
    self.db.add_documents(new_chunks, ids=[c.metadata["chunk_id"] for c in new_chunks])
```

---

## 2. Distance to Similarity Normalization

ChromaDB uses HNSW (Hierarchical Navigable Small World) index with squared $L_2$ Euclidean distance or cosine distance:
$$d = \|u - v\|_2^2 \quad \text{or} \quad d = 1 - \cos(u, v)$$

In distance metrics, **smaller values indicate higher semantic similarity** ($d = 0$ is an exact match).

### The Inverted Scoring Bug Fixed
In the original implementation, the system applied:
```python
if score >= score_threshold: # score_threshold was 0.6
```
Because Chroma returned distances (e.g., $0.15$ for a very close match), the comparison `0.15 >= 0.6` evaluated to `False`, discarding the most relevant chunks!

### The Modernized Solution
We convert raw distance into a monotonic normalized similarity score in the interval $(0, 1]$:
$$\text{similarity} = \frac{1}{1 + d}$$

- When $d = 0.0$ (identical): $\text{similarity} = 1.0$
- When $d = 0.5$ (very close): $\text{similarity} = 0.667$
- When $d = 2.0$ (distant): $\text{similarity} = 0.333$

```python
dist = float(raw_score)
similarity = 1.0 / (1.0 + dist)
if similarity >= score_threshold:
    filtered_results.append({
        "chunk_id": doc.metadata.get("chunk_id"),
        "content": doc.page_content,
        "score": round(similarity, 4),
        "distance": round(dist, 4),
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page", 0)
    })
```

---

## 3. Vector Space Visualization Mathematics (PCA)

Dense embeddings have high dimensionality (e.g., 384 dimensions from `all-MiniLM-L6-v2` or 1024 dimensions from `bge-m3`). Direct visual representation requires dimensionality reduction.

### Principal Component Analysis (PCA)
PCA projects the high-dimensional data matrix $X \in \mathbb{R}^{N \times D}$ into an optimal lower-dimensional orthogonal subspace $Z \in \mathbb{R}^{N \times K}$ ($K = 3$ for 3D and $K = 2$ for 2D) that maximizes the preserved variance.

1. **Mean-center the embeddings matrix**:
   $$\bar{X} = X - \mu$$
2. **Compute the Covariance Matrix**:
   $$\Sigma = \frac{1}{N - 1} \bar{X}^T \bar{X}$$
3. **Eigendecomposition**:
   $$\Sigma v_i = \lambda_i v_i$$
   where $v_i$ are the eigenvectors (principal axes) and $\lambda_i$ are the eigenvalues representing variance along each axis.
4. **Coordinate Projection**:
   Select the top 3 eigenvectors $W = [v_1, v_2, v_3] \in \mathbb{R}^{D \times 3}$:
   $$Z_{3D} = \bar{X} W$$

In [`backend/modules/visualizer.py`](file:///home/vicky/Projects/Local_MultiAgentic_RAG_System/backend/modules/visualizer.py):
```python
embeddings = np.array([c["embedding"] for c in valid_chunks])
pca_3d = PCA(n_components=min(3, num_samples, embeddings.shape[1]))
coords_3d = pca_3d.fit_transform(embeddings)

pca_2d = PCA(n_components=min(2, num_samples, embeddings.shape[1]))
coords_2d = pca_2d.fit_transform(embeddings)
```

---

## 4. Semantic Similarity Graph (Nearest-Neighbor Edges)

Points alone in 3D space show clusters, but do not show the relationship strength between chunks. Our visualizer computes a **cosine similarity graph**:

1. **Cosine Similarity Matrix**:
   $$S_{i,j} = \frac{e_i \cdot e_j}{\|e_i\|_2 \|e_j\|_2}$$
2. **Diagonal Suppression**: Set $S_{i,i} = 0$.
3. **Top-$K$ Neighbor Connection**:
   For each chunk $i$, identify the top $k = 3$ nearest chunks. If $S_{i,j} > 0.4$, add an edge line connecting $(x_i, y_i, z_i)$ and $(x_j, y_j, z_j)$.

```python
sim_matrix = cosine_similarity(embeddings)
np.fill_diagonal(sim_matrix, 0)
k = min(3, num_samples - 1)

for i in range(num_samples):
    top_k_idx = np.argsort(sim_matrix[i])[-k:]
    for j in top_k_idx:
        score = float(sim_matrix[i][j])
        if score > 0.4 and i < j:
            edges.append({
                "source_pos": [points[i]["x"], points[i]["y"], points[i]["z"]],
                "target_pos": [points[j]["x"], points[j]["y"], points[j]["z"]],
                "similarity": round(score, 3)
            })
```

---

## 5. WebGL Plotly 3D Scatter Rendering

The visualization is rendered in a WebGL-accelerated Plotly figure with:
- Distinct palette colors assigned to each PDF document source.
- Multi-line hover cards detailing source filename, page number, chunk ID, and a text preview.
- Semi-transparent network lines illustrating semantic bridges across document pages.
- Dark theme styling (`#0b0f19`).

---

## 6. Database Management Capabilities

Both the CLI (`manage_database.py`) and API routes (`/api/knowledge`) provide full operational control:

| Operation | CLI Option | REST Endpoint | Function |
| :--- | :--- | :--- | :--- |
| **Collection Stats** | Option 1 / `--stats` | `GET /api/knowledge/stats` | Chunks count, sources, page counts |
| **Inspect Chunks** | Option 2 | `GET /api/knowledge/chunks` | Layer-by-layer exploration |
| **Vector Search** | Option 3 | `GET /api/knowledge/search` | Direct similarity search |
| **Ingest / Refresh** | Option 4 / `--refresh`| `POST /api/knowledge/refresh` | Ingests PDFs from `knowledge_base/` |
| **Delete Document** | Option 5 | `DELETE /api/knowledge/file/{name}`| Removes all chunks for a file |
| **Delete Page** | Option 6 | N/A (CLI) | Removes chunks for a specific page |
| **Reset Collection**| Option 7 | `POST /api/knowledge/reset` | Clears all embeddings & resets HNSW |
| **Launch 3D Plot** | Option 8 / `--visualize`| `GET /api/knowledge/visualization` | Opens WebGL 3D scatter in browser |
