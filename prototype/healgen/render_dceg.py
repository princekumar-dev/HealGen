import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_enhanced_dceg(output_path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Title
    ax.text(5.0, 6.9, "Dynamic Claim--Evidence Graph (DCEG)", 
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1E293B")
    ax.text(5.0, 6.55, "Bipartite verification state: claims evaluated against retrieved evidence pool",
            ha="center", va="center", fontsize=8, color="#64748B", style="italic")

    # Column Headers
    ax.text(2.3, 6.15, "ATOMIC CLAIMS (Draft Assertions)", ha="center", va="center", 
            fontsize=8.5, fontweight="bold", color="#0F172A")
    ax.text(7.7, 6.15, "EVIDENCE PASSAGES (Pool)", ha="center", va="center", 
            fontsize=8.5, fontweight="bold", color="#0F172A")

    # Draw Nodes as Cards
    # c1: Supported
    c1_box = patches.FancyBboxPatch((0.4, 4.45), 3.8, 1.25, boxstyle="round,pad=0.1,rounding_size=0.15",
                                   ec="#16A34A", fc="#F0FDF4", lw=1.5)
    ax.add_patch(c1_box)
    ax.text(0.6, 5.4, "c1 [Supported]", fontsize=8, fontweight="bold", color="#15803D")
    ax.text(3.9, 5.4, "Risk: 0.12", fontsize=7.5, ha="right", color="#166534")
    ax.text(0.6, 4.85, "\"Douglas Corrigan was an\n American aviator.\"", fontsize=7.5, color="#1E293B")

    # c2: Supported
    c2_box = patches.FancyBboxPatch((0.4, 2.85), 3.8, 1.25, boxstyle="round,pad=0.1,rounding_size=0.15",
                                   ec="#16A34A", fc="#F0FDF4", lw=1.5)
    ax.add_patch(c2_box)
    ax.text(0.6, 3.8, "c2 [Supported]", fontsize=8, fontweight="bold", color="#15803D")
    ax.text(3.9, 3.8, "Risk: 0.25", fontsize=7.5, ha="right", color="#166534")
    ax.text(0.6, 3.25, "\"He was born in\n Galveston, Texas.\"", fontsize=7.5, color="#1E293B")

    # c3: Unsupported / Flagged
    c3_box = patches.FancyBboxPatch((0.4, 1.25), 3.8, 1.25, boxstyle="round,pad=0.1,rounding_size=0.15",
                                   ec="#DC2626", fc="#FEF2F2", lw=1.8, linestyle="--")
    ax.add_patch(c3_box)
    ax.text(0.6, 2.2, "c3 [FLAGGED FOR REPAIR]", fontsize=8, fontweight="bold", color="#B91C1C")
    ax.text(3.9, 2.2, "Risk: 0.89", fontsize=7.5, ha="right", color="#991B1B")
    ax.text(0.6, 1.65, "\"He flew non-stop to\n Tokyo in 1938.\"", fontsize=7.5, color="#1E293B")

    # e1: Evidence
    e1_box = patches.FancyBboxPatch((5.8, 3.75), 3.8, 1.95, boxstyle="round,pad=0.1,rounding_size=0.15",
                                   ec="#2563EB", fc="#EFF6FF", lw=1.4)
    ax.add_patch(e1_box)
    ax.text(6.0, 5.4, "e1 [Gold / Corpus Passage]", fontsize=8, fontweight="bold", color="#1D4ED8")
    ax.text(6.0, 4.5, "\"Douglas Corrigan (1907-1995)\nwas an American aviator\nborn in Galveston, Texas...\"",
            fontsize=7.5, color="#1E293B")

    # e2: Evidence
    e2_box = patches.FancyBboxPatch((5.8, 1.25), 3.8, 1.95, boxstyle="round,pad=0.1,rounding_size=0.15",
                                   ec="#2563EB", fc="#EFF6FF", lw=1.4)
    ax.add_patch(e2_box)
    ax.text(6.0, 2.9, "e2 [Retrieved BM25 Passage]", fontsize=8, fontweight="bold", color="#1D4ED8")
    ax.text(6.0, 2.0, "\"In July 1938, Corrigan flew\nnon-stop from NY to Dublin,\nIreland ('Wrong Way')...\"",
            fontsize=7.5, color="#1E293B")

    # Draw Directed Edges
    # e1 -> c1
    ax.annotate("", xy=(4.2, 5.1), xytext=(5.8, 4.9),
                arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=2.0, shrinkA=3, shrinkB=3))
    ax.text(5.0, 5.25, "supp: 0.88", fontsize=7, ha="center", va="bottom",
            fontweight="bold", color="#15803D",
            bbox=dict(boxstyle="round,pad=0.15", fc="#DCFCE7", ec="#86EFAC", lw=0.6))

    # e1 -> c2
    ax.annotate("", xy=(4.2, 3.5), xytext=(5.8, 4.3),
                arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=2.0, shrinkA=3, shrinkB=3))
    ax.text(5.0, 3.8, "supp: 0.96", fontsize=7, ha="center", va="bottom",
            fontweight="bold", color="#15803D",
            bbox=dict(boxstyle="round,pad=0.15", fc="#DCFCE7", ec="#86EFAC", lw=0.6))

    # e2 -> c3 (Contradiction / Low support)
    ax.annotate("", xy=(4.2, 1.85), xytext=(5.8, 2.1),
                arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=2.0, linestyle="--", shrinkA=3, shrinkB=3))
    ax.text(5.0, 2.15, "supp: 0.14 (contradicts)", fontsize=6.8, ha="center", va="bottom",
            fontweight="bold", color="#B91C1C",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FEE2E2", ec="#FCA5A5", lw=0.6))

    # Pedagogical Legend / Rule at the bottom
    rule_box = patches.FancyBboxPatch((0.4, 0.15), 9.2, 0.75, boxstyle="round,pad=0.1,rounding_size=0.1",
                                     ec="#CBD5E1", fc="#F8FAFC", lw=1.0)
    ax.add_patch(rule_box)
    ax.text(0.6, 0.55, "Decision Rule:", fontsize=7.5, fontweight="bold", color="#334155")
    ax.text(1.9, 0.55, "Claim supported if max(support) >= 0.50 without contradiction.",
            fontsize=7.2, color="#475569")
    ax.text(0.6, 0.28, "Repair Action:", fontsize=7.5, fontweight="bold", color="#991B1B")
    ax.text(1.9, 0.28, "c1, c2 preserved byte-identical. c3 flagged -> LSHE localized repair rewrites only Sentence 3.",
            fontsize=7.2, color="#991B1B", fontweight="semibold")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Rendered DCEG figure to {output_path}")

if __name__ == "__main__":
    out = Path("c:/Users/princ/Downloads/builder/verigen/paper/figures/dceg_example.png")
    draw_enhanced_dceg(out)
