"""Dynamic Claim-Evidence Graph (DCEG, Layer 6).

Maintains a graph of atomic claims, evidence passages, support links,
and contradiction relationships for localized self-healing decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from healgen import config


class DCEG:
    """Dynamic Claim-Evidence Graph for tracking claim validation state."""

    def __init__(self) -> None:
        self.g = nx.DiGraph()
        self.claims: dict[str, dict[str, Any]] = {}
        self.evidence: dict[str, dict[str, Any]] = {}
        self.contradictions: set[tuple[str, str]] = set()

    def add_claim(self, cid: str, text: str, risk: float = 0.0) -> None:
        """Add an atomic claim node."""
        cid = str(cid)
        self.claims[cid] = {"text": text, "risk": float(risk)}
        self.g.add_node(cid, node_type="claim", text=text, risk=float(risk))

    def add_evidence(
        self, eid: str, text: str, freshness: float = 1.0, weight: float = 0.0
    ) -> None:
        """Add an evidence passage node."""
        eid = str(eid)
        self.evidence[eid] = {
            "text": text,
            "freshness": float(freshness),
            "weight": float(weight),
        }
        self.g.add_node(
            eid,
            node_type="evidence",
            text=text,
            freshness=float(freshness),
            weight=float(weight),
        )

    def link(self, cid: str, eid: str, support: float) -> None:
        """Add a directed support edge from evidence to claim."""
        cid, eid = str(cid), str(eid)
        if cid not in self.claims:
            self.add_claim(cid, text="", risk=0.0)
        if eid not in self.evidence:
            self.add_evidence(eid, text="")
        self.g.add_edge(eid, cid, edge_type="support", support=float(support), weight=float(support))

    def add_contradiction(self, cid1: str, cid2: str) -> None:
        """Add an undirected contradiction edge between two claims."""
        cid1, cid2 = str(cid1), str(cid2)
        pair = (min(cid1, cid2), max(cid1, cid2))
        self.contradictions.add(pair)
        self.g.add_edge(cid1, cid2, edge_type="contradiction")
        self.g.add_edge(cid2, cid1, edge_type="contradiction")

    def unsupported(self) -> list[str]:
        """Return IDs of claims that lack adequate support (>=0.5) or have contradictions.

        A claim is unsupported if:
        1. It has no incoming support edge from any evidence with support >= 0.5.
        2. Or it is involved in any active contradiction edge.
        """
        unsupported_cids: list[str] = []
        for cid in self.claims:
            # Check contradiction involvement
            in_contradiction = any(cid in pair for pair in self.contradictions)
            if in_contradiction:
                unsupported_cids.append(cid)
                continue

            # Check support edges
            support_scores = [
                data.get("support", 0.0)
                for u, v, data in self.g.in_edges(cid, data=True)
                if data.get("edge_type") == "support"
            ]
            if not support_scores or max(support_scores) < 0.5:
                unsupported_cids.append(cid)

        return unsupported_cids

    def to_dot(self) -> str:
        """Generate a Graphviz DOT representation of the graph."""
        lines = ["digraph DCEG {", '  rankdir="LR";', "  node [fontname=\"sans-serif\"];"]

        # Claim nodes
        for cid, meta in self.claims.items():
            label = f"{cid}\\nrisk: {meta['risk']:.2f}"
            lines.append(f'  "{cid}" [shape=box, style=filled, fillcolor="#E1F5FE", label="{label}"];')

        # Evidence nodes
        for eid, meta in self.evidence.items():
            label = f"{eid}\\nfreshness: {meta['freshness']:.2f}"
            lines.append(f'  "{eid}" [shape=ellipse, style=filled, fillcolor="#E8F5E9", label="{label}"];')

        # Edges
        for u, v, data in self.g.edges(data=True):
            etype = data.get("edge_type", "support")
            if etype == "support":
                supp = data.get("support", 0.0)
                lines.append(f'  "{u}" -> "{v}" [label="supp: {supp:.2f}", color="#2E7D32"];')
            elif etype == "contradiction" and u < v:  # avoid duplicate undirected in dot
                lines.append(f'  "{u}" -> "{v}" [dir=both, style=dashed, color="#C62828", label="contradiction"];')

        lines.append("}")
        return "\n".join(lines)

    def draw(self, path: Path | str) -> None:
        """Draw the graph using spring layout and save as PNG."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        pos = nx.spring_layout(self.g, seed=42)

        # Node colors
        claim_nodes = [n for n in self.g.nodes if n in self.claims]
        ev_nodes = [n for n in self.g.nodes if n in self.evidence]

        nx.draw_networkx_nodes(
            self.g, pos, nodelist=claim_nodes, node_color="#90CAF9", node_shape="s", node_size=1500, ax=ax
        )
        nx.draw_networkx_nodes(
            self.g, pos, nodelist=ev_nodes, node_color="#A5D6A7", node_shape="o", node_size=1500, ax=ax
        )
        nx.draw_networkx_labels(self.g, pos, font_size=9, font_family="sans-serif", ax=ax)

        # Edge colors
        support_edges = [
            (u, v) for u, v, d in self.g.edges(data=True) if d.get("edge_type") == "support"
        ]
        contra_edges = [
            (u, v) for u, v, d in self.g.edges(data=True) if d.get("edge_type") == "contradiction"
        ]

        nx.draw_networkx_edges(
            self.g, pos, edgelist=support_edges, edge_color="#2E7D32", arrows=True, ax=ax
        )
        nx.draw_networkx_edges(
            self.g, pos, edgelist=contra_edges, edge_color="#C62828", style="dashed", ax=ax
        )

        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def export_example_figure() -> Path:
    """Generate and export the canonical DCEG figure for the paper."""
    fig_path = config.PATHS["figures"] / "dceg_example.png"
    paper_fig_path = config.PROTOTYPE_ROOT.parent / "paper" / "figures" / "dceg_example.png"
    
    # Import and render pedagogical DCEG figure
    from healgen.render_dceg import draw_enhanced_dceg
    draw_enhanced_dceg(fig_path)
    if paper_fig_path.parent.exists():
        draw_enhanced_dceg(paper_fig_path)
    return fig_path
