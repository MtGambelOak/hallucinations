"""
Print top-k results from saved SAE direction analyses.

Loops over all sae_analysis_*.json in results/ and writes a combined
report to results/sae_analysis_report.txt (and stdout).

Usage:
    python print_sae_analysis.py
    python print_sae_analysis.py --analysis results/sae_analysis_rm_sae_helpsteer2_factuality.json
    python print_sae_analysis.py --out results/my_report.txt
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path


def print_one(data, labels, args, out):
    attrs  = data["attributes"]
    d_sae  = data["d_sae"]
    C      = np.array(data.get("dot_products", data.get("contributions")))  # (d_sae, n_attr)

    def latent_str(j):
        lbl = labels.get(j, "")
        return f"latent {j:3d}" + (f"  [{lbl}]" if lbl else "")

    def w(s):
        out.write(s + "\n")

    w(f"\n{'='*70}")
    w(f"  SAE: {data['sae_path']}")
    w(f"  d_sae={d_sae}  n_attr={data['n_attr']}")
    w(f"{'='*70}")

    # Per-latent: top attributes
    w(f"\n{'='*70}")
    w(f"  TOP {args.top_attr} ArmoRM ATTRIBUTES PER SAE LATENT")
    w(f"{'='*70}")
    for j in range(min(args.max_latents, d_sae)):
        top = np.argsort(C[j])[::-1][:args.top_attr]
        w(f"  {latent_str(j)}")
        for i in top:
            w(f"      {attrs[i]}  cos={C[j,i]:+.4f}")

    # Per-attribute: top latents
    w(f"\n{'='*70}")
    w(f"  TOP {args.top_lat} SAE LATENTS PER ArmoRM ATTRIBUTE")
    w(f"{'='*70}")
    for i, attr in enumerate(attrs):
        top = np.argsort(C[:, i])[::-1][:args.top_lat]
        w(f"\n  {attr}")
        w(f"  {'-'*60}")
        for j in top:
            w(f"    {latent_str(j)}  cos={C[j,i]:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default=None,
                        help="Single analysis file (default: all in results/)")
    parser.add_argument("--labels",   default=None,
                        help="Labels file (only with --analysis)")
    parser.add_argument("--out",      default="results/sae_analysis_report.txt",
                        help="Output txt file")
    parser.add_argument("--top_attr", type=int, default=3)
    parser.add_argument("--max_latents", type=int, default=64)
    parser.add_argument("--top_lat",  type=int, default=10)
    args = parser.parse_args()

    if args.analysis:
        analysis_files = [Path(args.analysis)]
    else:
        analysis_files = sorted(Path("results").glob("sae_analysis_*.json"))
        if not analysis_files:
            print("No sae_analysis_*.json found in results/")
            return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        # tee to both stdout and file
        class Tee:
            def write(self, s): f.write(s); sys.stdout.write(s)
            def flush(self): f.flush(); sys.stdout.flush()
        out = Tee()

        for analysis_path in analysis_files:
            with open(analysis_path) as af:
                data = json.load(af)

            # auto-detect labels
            labels = {}
            label_path = args.labels
            if label_path is None:
                stem = analysis_path.stem.replace("sae_analysis_", "sae_labels_")
                candidate = Path("results") / f"{stem}.json"
                if candidate.exists():
                    label_path = str(candidate)
            if label_path and Path(label_path).exists():
                with open(label_path) as lf:
                    raw = json.load(lf)
                labels = {int(k): v["label"] for k, v in raw.items()}
                label_path = None  # reset for next file

            print_one(data, labels, args, out)

    print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
