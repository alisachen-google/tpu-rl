#!/usr/bin/env python3
"""Render the Mermaid flowchart with a fixed, publication-friendly layout.

Requires matplotlib. Labels, colors, edge styles, and edge captions come from
the adjacent .mmd file; coordinates below keep both compiler paths vertical.
Run from any directory: python3 docs/diagrams/render_tpu_communication.py
"""

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
STEM = "tpu-communication-flow"
WIDTH, HEIGHT, DPI = 2400, 1380, 100
SOURCE = (HERE / f"{STEM}.mmd").read_text(encoding="utf-8")
NODES = {
    node: (label, style)
    for node, label, style in re.findall(
        r'^\s*(\w+)\["(.*?)"\]:::(\w+)\s*$', SOURCE, re.MULTILINE
    )
}
GROUPS = dict(re.findall(r'subgraph (\w+)\["(.*?)"\]', SOURCE))
STYLES = {
    name: dict(item.split(":", 1) for item in properties.split(","))
    for name, properties in re.findall(r"classDef (\w+) (.*?);", SOURCE)
}
EDGES = {}
for source_line in SOURCE.splitlines():
    if not re.match(r"^\s*\w+\s+(-->|-\.->)", source_line):
        continue
    labels = re.findall(r"\|([^|]+)\|", source_line)
    parts = re.split(r"\s*(-->|-\.->)\s*", re.sub(r"\|[^|]+\|", "", source_line.strip()))
    for index in range(0, len(parts) - 2, 2):
        EDGES[parts[index], parts[index + 2]] = (
            parts[index + 1] == "-.->", labels[0] if labels else ""
        )

# Pixel coordinates: x, y, width, height. The canvas origin is at top left.
PANELS = {
    "Authoring": (48, 196, 380, 1080),
    "HLOPath": (500, 196, 560, 520),
    "LLOPath": (500, 756, 560, 520),
    "LLOLayer": (1152, 478, 480, 428),
    "RoutingAndHardware": (1724, 196, 628, 1080),
}
BOXES = {
    "M1": (76, 303, 324, 142),
    "M2": (76, 838, 324, 142),
    "H1": (528, 326, 504, 96),
    "H2": (528, 460, 504, 104),
    "H3": (528, 602, 504, 102),
    "P1": (528, 854, 504, 110),
    "P2": (528, 1002, 504, 104),
    "P3": (528, 1144, 504, 106),
    "L1": (1180, 594, 424, 118),
    "L2": (1180, 762, 424, 118),
    "HW1": (1752, 746, 524, 150),
    "RA": (1752, 430, 524, 146),
    "RT": (1752, 986, 524, 110),
    "NET": (1752, 1150, 524, 100),
}
ROUTES = {
    ("M1", "H1"): [(400, 374), (528, 374)],
    ("M2", "P1"): [(400, 909), (528, 909)],
    ("H1", "H2"): [(780, 422), (780, 460)],
    ("H2", "H3"): [(780, 564), (780, 602)],
    ("P1", "P2"): [(780, 964), (780, 1002)],
    ("P2", "P3"): [(780, 1106), (780, 1144)],
    ("H3", "L1"): [(1032, 653), (1180, 653)],
    ("P3", "L1"): [(1032, 1197), (1108, 1197), (1108, 688), (1180, 688)],
    ("L1", "L2"): [(1392, 712), (1392, 762)],
    ("L2", "HW1"): [(1604, 821), (1752, 821)],
    ("HW1", "RT"): [(2014, 896), (2014, 986)],
    ("RA", "RT"): [(2014, 576), (2014, 616), (2312, 616), (2312, 1041), (2276, 1041)],
    ("RT", "NET"): [(2014, 1096), (2014, 1150)],
}
if set(NODES) != set(BOXES) or set(EDGES) != set(ROUTES) or set(GROUPS) != set(PANELS):
    raise ValueError("Flowchart structure changed; update the corresponding layout coordinates.")

# Wrap long source lines without changing their words or punctuation.
WRAPS = {
    "M1": [("all_gather, ", "all_gather,\n")],
    "M2": [("pallas_call(), ", "pallas_call(),\n")],
    "P2": [("tpu_custom_call (", "tpu_custom_call\n(")],
    "L1": [("::", "::\n")],
    "HW1": [("dma_type, ", "dma_type,\n")],
    "NET": [("->", "→")],
}
PANEL_WRAPS = {
    "Authoring": [(" Layer", "\nLayer")],
    "LLOPath": [(" (Pallas", "\n(Pallas")],
    "LLOLayer": [(" (LLO)", "\n(LLO)")],
    "RoutingAndHardware": [(" Routing", "\nRouting")],
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "path",  # Keep exports independent of installed viewer fonts.
    "svg.hashsalt": STEM,
    "pdf.fonttype": 42,
})
fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor="white")
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
ax.set_axis_off()
text_bounds = []


def text(x, y, value, size=24, *, color="#000000", bold=False, mono=False,
         align="center", bounds=None, gid=None):
    artist = ax.text(
        x, y, value, ha=align, va="center", color=color, fontsize=size * 72 / DPI,
        fontweight="bold" if bold else "normal",
        fontfamily="DejaVu Sans Mono" if mono else "DejaVu Sans",
        zorder=5, gid=gid,
    )
    if bounds:
        text_bounds.append((artist, bounds))
    return artist


def rounded_box(bounds, fill, stroke, width=1.5, radius=8, gid=None, zorder=3):
    x, y, w, h = bounds
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=stroke, linewidth=width * 72 / DPI,
        zorder=zorder, gid=gid,
    )
    ax.add_patch(box)


def arrow(points, *, dashed=False, gid=None):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=13,
        linewidth=2 * 72 / DPI, color="#333333",
        linestyle=(0, (5, 4)) if dashed else "solid",
        capstyle="round", joinstyle="round", zorder=2, gid=gid,
    )
    ax.add_patch(patch)


text(48, 66, "TPU communication lowering & routing", 43, bold=True, align="left")
text(48, 112, "Standard XLA HLO and custom kernel paths converge at the low-level optimizer.",
     23, color="#555555", align="left")
ax.plot([48, 2352], [152, 152], color="#cccccc", linewidth=0.9, zorder=0)

for group, bounds in PANELS.items():
    rounded_box(bounds, "#fafafa", "#b8b8b8", radius=12, gid=f"panel-{group}", zorder=0)
    heading = GROUPS[group]
    for old, new in PANEL_WRAPS.get(group, []):
        heading = heading.replace(old, new)
    x, y, w, h = bounds
    for index, line in enumerate(heading.splitlines()):
        text(x + 28, y + 40 + 32 * index, line, 25, bold=True, align="left",
             bounds=bounds, gid=f"heading-{group}-{index}")

for (start, end), points in ROUTES.items():
    dashed, _ = EDGES[start, end]
    arrow(points, dashed=dashed, gid=f"edge-{start}-{end}")

for node, bounds in BOXES.items():
    label, class_name = NODES[node]
    style = STYLES[class_name]
    rounded_box(bounds, style["fill"], style["stroke"],
                float(style["stroke-width"].removesuffix("px")), gid=f"node-{node}")
    label = label.replace("<br>", "\n")
    for old, new in WRAPS.get(node, []):
        label = label.replace(old, new)
    lines = label.splitlines()
    x, y, w, h = bounds
    spacing = 34 if len(lines) == 3 else 39
    if node == "P2":
        spacing = 28
    for index, line in enumerate(lines):
        code_node = node in {"M1", "M2", "H2", "H3", "P1", "P2", "HW1"}
        mono = node == "L1" or (index > 0 and code_node and not line.startswith("("))
        size = 21 if mono else 24
        if index == 0 and node != "L1":
            size = 22 if node == "M2" else 25
        if node == "P2" and index == 2:
            size = 19
        if node == "H3" and index == 1:
            size = 20
        if node == "NET":
            size = 24 if index == 0 else 23
        if node == "L2":
            size = 24
        text(x + w / 2, y + h / 2 + (index - (len(lines) - 1) / 2) * spacing,
             line, size, color=style["color"], bold=index == 0 and node != "L1",
             mono=mono, bounds=bounds, gid=f"label-{node}-{index}")

text(2040, 941, EDGES["HW1", "RT"][1], 21, align="left")
text(2170, 595, EDGES["RA", "RT"][1], 21)
arrow([(48, 1326), (111, 1326)])
text(128, 1326, "Lowering / execution flow", 20, color="#555555", align="left")
arrow([(500, 1326), (563, 1326)], dashed=True)
text(580, 1326, "Routing-table population", 20, color="#555555", align="left")

# Fail before exporting if labels do not fit the fixed layout.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist, (x, y, w, h) in text_bounds:
    bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    left, right = sorted([bounds.x0, bounds.x1])
    top, bottom = sorted([bounds.y0, bounds.y1])
    if not (x + 8 <= left and right <= x + w - 8 and y + 8 <= top and bottom <= y + h - 8):
        raise ValueError(f"Label exceeds its box: {artist.get_text()!r}")

title = "TPU communication lowering and routing"
description = " ".join(re.sub(r"<br>", " ", label) for label, _ in NODES.values())
fig.savefig(HERE / f"{STEM}.svg", metadata={"Title": title, "Date": None})
fig.savefig(HERE / f"{STEM}.png", dpi=200, metadata={"Title": title})
fig.savefig(HERE / f"{STEM}.pdf", metadata={"Title": title, "CreationDate": None, "ModDate": None})
plt.close(fig)

# Add accessible text to the SVG, whose visible glyphs are portable vector paths.
svg_path = HERE / f"{STEM}.svg"
svg = svg_path.read_text(encoding="utf-8")
start = svg.index("<svg ")
end = svg.index(">", start)
svg = svg[:end] + ' role="img" aria-labelledby="diagram-title diagram-description"' + svg[end:]
end = svg.index(">", start)
title_element = ET.Element("title", {"id": "diagram-title"})
title_element.text = title
desc_element = ET.Element("desc", {"id": "diagram-description"})
desc_element.text = (
    "Two compilation paths converge at LLO and feed hardware DMA, routing, and the network. "
    "Routing algorithms populate on-chip routing tables using a dashed connection. " + description
)
svg = svg[:end + 1] + ET.tostring(title_element, encoding="unicode") + ET.tostring(desc_element, encoding="unicode") + svg[end + 1:]
svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
ET.fromstring(svg)
svg_path.write_text(svg, encoding="utf-8")
print(f"Rendered {len(NODES)} nodes and {len(EDGES)} connections: SVG, PNG (4800×2760), PDF.")
