"""Regenerate the fictional sample resumes in samples/resumes/ (no real people; deterministic)."""
from pathlib import Path

from . import heldout as ho
from . import synthetic as syn

OUT = Path(__file__).resolve().parents[1] / "samples" / "resumes"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cands, _ = ho.build(4242, per_arch=[("exact", 1), ("paraphrased", 1), ("project", 1), ("near_miss", 1), ("stuffed", 1)], layouts=("L1", "L3", "L5", "L6", "L2"))
    picks = [c for c in cands if c.target == "H1"]
    names = {"exact": "strong_match", "paraphrased": "paraphrased_skills", "project": "student_projects", "near_miss": "missing_one_essential", "stuffed": "keyword_stuffed"}
    for c in picks:
        fn, data = ho.render_cand(c)
        (OUT / f"{names[c.archetype]}{Path(fn).suffix}").write_bytes(data)
    scan = next(c for c in picks if c.layout != "L3")
    (OUT / "scanned_image_only.pdf").write_bytes(syn.render_scanned_pdf([x for x in scan.structure if x[0] != "col"], dpi=150, jpeg=True))
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
