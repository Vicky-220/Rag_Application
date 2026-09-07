"""
Vector Visualization Module
Projects high-dimensional vector embeddings into 3D/2D space using PCA,
computes semantic similarity networks, and generates interactive Plotly visualizations.
"""
import os
from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from backend.modules.vector_db import VectorStore


class VectorVisualizer:
    """
    Generates 2D and 3D vector space visualizations of knowledge base chunks.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Extract chunk embeddings, reduce dimensionality via PCA, and compute graph edges.
        
        Returns:
            Dict containing 2D/3D points, metadata, and similarity edges.
        """
        chunks = self.vector_store.get_all_chunks()
        if not chunks:
            return {"total_chunks": 0, "points": [], "edges": [], "sources": []}

        # Filter chunks that actually have embeddings
        valid_chunks = [c for c in chunks if c.get("embedding") is not None and len(c["embedding"]) > 0]
        if not valid_chunks:
            return {"total_chunks": 0, "points": [], "edges": [], "sources": []}

        embeddings = np.array([c["embedding"] for c in valid_chunks])
        num_samples = len(embeddings)

        # Compute 2D and 3D PCA
        n_comp_3d = min(3, num_samples, embeddings.shape[1])
        n_comp_2d = min(2, num_samples, embeddings.shape[1])

        pca_3d = PCA(n_components=n_comp_3d)
        coords_3d = pca_3d.fit_transform(embeddings)

        pca_2d = PCA(n_components=n_comp_2d)
        coords_2d = pca_2d.fit_transform(embeddings)

        points = []
        sources = sorted(list(set(c["metadata"].get("source", "Unknown") for c in valid_chunks)))

        for i, c in enumerate(valid_chunks):
            x = float(coords_3d[i, 0]) if n_comp_3d >= 1 else 0.0
            y = float(coords_3d[i, 1]) if n_comp_3d >= 2 else 0.0
            z = float(coords_3d[i, 2]) if n_comp_3d >= 3 else 0.0

            x2d = float(coords_2d[i, 0]) if n_comp_2d >= 1 else 0.0
            y2d = float(coords_2d[i, 1]) if n_comp_2d >= 2 else 0.0

            source_file = c["metadata"].get("source", "Unknown")
            page = c["metadata"].get("page", 0)
            chunk_id = c["metadata"].get("chunk_id", c["id"])
            preview = c["content"][:250] + ("..." if len(c["content"]) > 250 else "")

            points.append({
                "id": chunk_id,
                "source": os.path.basename(source_file),
                "full_source": source_file,
                "page": page,
                "x": x,
                "y": y,
                "z": z,
                "x2d": x2d,
                "y2d": y2d,
                "preview": preview,
            })

        # Compute cosine similarity edges between top-k neighbors
        edges = []
        if num_samples > 1:
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, 0)
            k = min(3, num_samples - 1)

            for i in range(num_samples):
                top_k_idx = np.argsort(sim_matrix[i])[-k:]
                for j in top_k_idx:
                    score = float(sim_matrix[i][j])
                    if score > 0.4 and i < j:
                        edges.append({
                            "source_id": points[i]["id"],
                            "target_id": points[j]["id"],
                            "source_pos": [points[i]["x"], points[i]["y"], points[i]["z"]],
                            "target_pos": [points[j]["x"], points[j]["y"], points[j]["z"]],
                            "similarity": round(score, 3)
                        })

        return {
            "total_chunks": len(points),
            "points": points,
            "edges": edges,
            "sources": [os.path.basename(s) for s in sources],
        }

    def generate_html_plot(self) -> str:
        """
        Build an interactive Plotly 3D scatter figure with dark cyberpunk theme.
        
        Returns:
            HTML string with embedded Plotly.js visualization.
        """
        data = self.get_visualization_data()
        points = data.get("points", [])

        if not points:
            return """
            <html>
            <head><title>Vector Visualization</title></head>
            <body style="background-color:#0b0f19; color:#e2e8f0; font-family:sans-serif; text-align:center; padding-top:100px;">
                <h2>No Vector Embeddings Found</h2>
                <p>Upload documents to the knowledge base to visualize chunk embeddings.</p>
            </body>
            </html>
            """

        fig = go.Figure()

        # Group points by source file for multi-colored clusters
        unique_sources = sorted(list(set(p["source"] for p in points)))
        color_palette = [
            "#38bdf8", "#ec4899", "#10b981", "#f59e0b",
            "#8b5cf6", "#06b6d4", "#f43f5e", "#84cc16"
        ]

        for idx, src in enumerate(unique_sources):
            src_points = [p for p in points if p["source"] == src]
            color = color_palette[idx % len(color_palette)]

            fig.add_trace(go.Scatter3d(
                x=[p["x"] for p in src_points],
                y=[p["y"] for p in src_points],
                z=[p["z"] for p in src_points],
                mode='markers',
                name=src,
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.9,
                    line=dict(width=1, color="#ffffff")
                ),
                text=[
                    f"<b>File:</b> {p['source']}<br>"
                    f"<b>Page:</b> {p['page']}<br>"
                    f"<b>Chunk ID:</b> {p['id']}<br><br>"
                    f"<b>Content:</b><br>{p['preview']}"
                    for p in src_points
                ],
                hoverinfo='text'
            ))

        # Add similarity edges as lines
        edges = data.get("edges", [])
        edge_x, edge_y, edge_z = [], [], []
        for e in edges:
            edge_x.extend([e["source_pos"][0], e["target_pos"][0], None])
            edge_y.extend([e["source_pos"][1], e["target_pos"][1], None])
            edge_z.extend([e["source_pos"][2], e["target_pos"][2], None])

        if edge_x:
            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                name='Semantic Connections',
                line=dict(color='rgba(148, 163, 184, 0.25)', width=1.5),
                hoverinfo='none',
                showlegend=True
            ))

        fig.update_layout(
            title=dict(
                text=f"Vector Space 3D Projection (Total Chunks: {len(points)})",
                font=dict(size=18, color="#f8fafc")
            ),
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            legend=dict(
                font=dict(color="#cbd5e1"),
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor="#334155",
                borderwidth=1
            ),
            scene=dict(
                xaxis=dict(
                    title="PCA 1",
                    color="#94a3b8",
                    gridcolor="#1e293b",
                    showbackground=True,
                    backgroundcolor="#0f172a"
                ),
                yaxis=dict(
                    title="PCA 2",
                    color="#94a3b8",
                    gridcolor="#1e293b",
                    showbackground=True,
                    backgroundcolor="#0f172a"
                ),
                zaxis=dict(
                    title="PCA 3",
                    color="#94a3b8",
                    gridcolor="#1e293b",
                    showbackground=True,
                    backgroundcolor="#0f172a"
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return pio.to_html(fig, include_plotlyjs='cdn', full_html=True)
