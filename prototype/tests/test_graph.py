"""Unit tests for the Dynamic Claim-Evidence Graph (DCEG, Layer 6)."""

from pathlib import Path
import pytest
from healgen.graph import DCEG


def test_unsupported_detection():
    """Claim with only weak link is unsupported; claim with strong link is supported."""
    graph = DCEG()
    graph.add_claim("c1", "Paris is in France.", risk=0.1)
    graph.add_claim("c2", "Paris is in Germany.", risk=0.9)
    graph.add_evidence("e1", "Paris is the capital of France.")

    # c1 is supported strongly
    graph.link("c1", "e1", support=0.95)
    # c2 is only weakly linked
    graph.link("c2", "e1", support=0.10)

    unsupported = graph.unsupported()
    assert "c2" in unsupported
    assert "c1" not in unsupported


def test_contradiction_flag():
    """Contradiction flags a claim as unsupported even if it has a supporting edge."""
    graph = DCEG()
    graph.add_claim("c1", "The event happened in 1990.", risk=0.2)
    graph.add_claim("c2", "The event happened in 2005.", risk=0.8)
    graph.add_evidence("e1", "The event took place in 1990.")

    graph.link("c1", "e1", support=0.9)
    assert graph.unsupported() == ["c2"]

    # Adding contradiction between c1 and c2 causes contradiction flag
    graph.add_contradiction("c1", "c2")
    unsupported = graph.unsupported()
    assert "c1" in unsupported
    assert "c2" in unsupported


def test_dot_export_contains_nodes():
    """DOT format export includes all nodes and edges."""
    graph = DCEG()
    graph.add_claim("c1", "Claim one.", risk=0.3)
    graph.add_evidence("e1", "Evidence one.")
    graph.link("c1", "e1", support=0.85)

    dot = graph.to_dot()
    assert "digraph" in dot.lower() or "graph" in dot.lower()
    assert "c1" in dot
    assert "e1" in dot
    assert "0.85" in dot or "support" in dot.lower()


def test_draw_writes_png(tmp_path: Path):
    """Drawing the graph saves a non-empty PNG file to disk."""
    graph = DCEG()
    graph.add_claim("c1", "Sample claim", risk=0.2)
    graph.add_evidence("e1", "Sample evidence")
    graph.link("c1", "e1", support=0.9)

    out_png = tmp_path / "test_graph.png"
    graph.draw(out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0
