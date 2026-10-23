"""Convert a KiCad board (.kicad_pcb, KiCad 6/7/8/9) into the board-map files:

  pos.csv   Ref,Val,Package,PosX,PosY,Rot,Side      (component centres, mm)
  pads.csv  Ref,Pin,X,Y,Net                          (absolute pad centres, mm)

Usage:
  python -m tools.kicad_to_csv path/to/board.kicad_pcb kb/boards/<board_id>
"""
import csv
import math
import sys
from pathlib import Path


def parse_sexpr(text):
    """Tiny S-expression parser -> nested lists of strings."""
    stack, cur, tok, i, n = [], [], "", 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    j += 1
                buf.append(text[j])
                j += 1
            cur.append("".join(buf))
            i = j + 1
            continue
        if c == "(":
            if tok:
                cur.append(tok); tok = ""
            stack.append(cur); cur = []
        elif c == ")":
            if tok:
                cur.append(tok); tok = ""
            done = cur; cur = stack.pop(); cur.append(done)
        elif c.isspace():
            if tok:
                cur.append(tok); tok = ""
        else:
            tok += c
        i += 1
    return cur[0] if cur else []


def children(node, name):
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == name]


def first(node, name):
    found = children(node, name)
    return found[0] if found else None


def _at(node):
    a = first(node, "at")
    if not a:
        return 0.0, 0.0, 0.0
    x, y = float(a[1]), float(a[2])
    rot = float(a[3]) if len(a) > 3 and not isinstance(a[3], list) else 0.0
    return x, y, rot


def _prop(fp, name):
    for p in children(fp, "property"):          # KiCad 7+
        if len(p) > 2 and p[1] == name:
            return p[2]
    for t in children(fp, "fp_text"):           # KiCad 6
        if len(t) > 2 and t[1] == name.lower():
            return t[2]
    return ""


def convert(pcb_path, out_dir):
    tree = parse_sexpr(Path(pcb_path).read_text(encoding="utf-8"))
    parts, pads = [], []
    for fp in children(tree, "footprint") + children(tree, "module"):
        ref, val = _prop(fp, "Reference"), _prop(fp, "Value")
        if not ref or ref.startswith("#"):
            continue
        package = fp[1].split(":")[-1] if len(fp) > 1 and isinstance(fp[1], str) else ""
        layer = first(fp, "layer")
        side = "bottom" if layer and "B." in layer[1] else "top"
        fx, fy, frot = _at(fp)
        parts.append([ref, val, package, round(fx, 4), round(fy, 4), frot, side])
        th = math.radians(frot)
        for pad in children(fp, "pad"):
            pin = pad[1] if len(pad) > 1 else ""
            if pin in ("", '""'):
                continue                         # mounting holes etc.
            px, py, _ = _at(pad)
            # KiCad: Y points down, positive rotation is counter-clockwise on screen
            ax = fx + px * math.cos(th) + py * math.sin(th)
            ay = fy - px * math.sin(th) + py * math.cos(th)
            net = first(pad, "net")
            netname = net[2] if net and len(net) > 2 else (net[1] if net and len(net) > 1 else "")
            pads.append([ref, pin, round(ax, 4), round(ay, 4), netname])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ref", "Val", "Package", "PosX", "PosY", "Rot", "Side"])
        w.writerows(sorted(parts))
    with open(out / "pads.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ref", "Pin", "X", "Y", "Net"])
        w.writerows(sorted(pads, key=lambda r: (r[0], str(r[1]).zfill(4))))
    return len(parts), len(pads)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m tools.kicad_to_csv board.kicad_pcb kb/boards/<board_id>")
    n_parts, n_pads = convert(sys.argv[1], sys.argv[2])
    print(f"wrote {n_parts} components and {n_pads} pads to {sys.argv[2]}")
