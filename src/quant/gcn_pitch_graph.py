"""
gcn_pitch_graph.py – Graph Convolutional Network ile saha grafiği analizi.
PyTorch Geometric: sahayı grafik ağı olarak modeller, oyuncu koordinasyonunu çözer.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("torch_geometric yüklü değil – GCN basit modda.")


if TORCH_AVAILABLE and PYG_AVAILABLE:
    class PitchGCN(nn.Module):
        """Saha grafiği üzerinde GCN."""
        def __init__(self, node_features: int = 4, hidden: int = 32, out_dim: int = 3):
            super().__init__()
            self.conv1 = GCNConv(node_features, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.fc = nn.Linear(hidden, out_dim)
            self.relu = nn.ReLU()

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.relu(self.conv1(x, edge_index))
            x = self.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.fc(x)


class GCNPitchGraph:
    """Saha grafik ağı analizi – oyuncu koordinasyon tespiti."""

    def __init__(self, model_path: str | None = None):
        self._model = None
        if TORCH_AVAILABLE and PYG_AVAILABLE:
            self._model = PitchGCN()
            self._model.eval()
            if model_path:
                try:
                    self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
                except Exception as e:
                    logger.warning(f"GCN model yükleme hatası: {e}")
        logger.debug("GCNPitchGraph başlatıldı.")

    def build_graph(self, player_positions: list[tuple[float, float]], teams: list[int]) -> dict:
        """Oyuncu pozisyonlarından graf oluşturur."""
        n = len(player_positions)
        if n < 2:
            return {"nodes": n, "edges": 0}

        positions = np.array(player_positions)
        team_ids = np.array(teams)

        # Edge'ler: aynı takımdaki yakın oyuncular arasında
        edges_src, edges_dst = [], []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                same_team = team_ids[i] == team_ids[j]
                threshold = 30.0 if same_team else 15.0
                if dist < threshold:
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])

        # Node features: x, y, team, distance_to_center
        center = np.array([52.5, 34.0])  # Saha merkezi
        dist_to_center = np.linalg.norm(positions - center, axis=1)
        node_features = np.column_stack([positions, team_ids, dist_to_center])

        if TORCH_AVAILABLE and PYG_AVAILABLE:
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long) if edges_src else torch.zeros(2, 0, dtype=torch.long),
                batch=torch.zeros(n, dtype=torch.long),
            )
            return {"data": data, "nodes": n, "edges": len(edges_src) // 2}

        return {
            "nodes": n,
            "edges": len(edges_src) // 2,
            "features": node_features,
            "edge_index": (edges_src, edges_dst),
        }

    def analyze_coordination(self, player_positions: list[tuple[float, float]], teams: list[int]) -> dict:
        """Takım koordinasyonunu analiz eder."""
        if len(player_positions) < 4:
            return {"home_coordination": 0.5, "away_coordination": 0.5}

        positions = np.array(player_positions)
        team_ids = np.array(teams)

        metrics = {}
        for team_id, team_name in [(0, "home"), (1, "away")]:
            mask = team_ids == team_id
            if mask.sum() < 2:
                metrics[f"{team_name}_coordination"] = 0.5
                continue

            team_pos = positions[mask]

            # Kompaktlık: oyuncular arası ortalama mesafe
            dists = []
            for i in range(len(team_pos)):
                for j in range(i + 1, len(team_pos)):
                    dists.append(np.linalg.norm(team_pos[i] - team_pos[j]))
            avg_dist = np.mean(dists) if dists else 50.0

            # Width ve Depth
            width = np.ptp(team_pos[:, 0]) if len(team_pos) > 1 else 0
            depth = np.ptp(team_pos[:, 1]) if len(team_pos) > 1 else 0

            # Koordinasyon skoru: kompaktlık + denge
            compactness = 1.0 / (1.0 + avg_dist / 20.0)
            balance = 1.0 - abs(width - depth) / (width + depth + 1e-6)
            coordination = 0.6 * compactness + 0.4 * balance

            metrics[f"{team_name}_coordination"] = float(np.clip(coordination, 0, 1))
            metrics[f"{team_name}_compactness"] = float(compactness)
            metrics[f"{team_name}_width"] = float(width)
            metrics[f"{team_name}_depth"] = float(depth)

        return metrics
