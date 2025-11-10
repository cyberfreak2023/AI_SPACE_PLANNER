# app.py
"""
AI Space Planner — Single Best Intelligent Layout with Setbacks, Doors & Windows
Run:
    python app.py
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import gradio as gr

# -----------------------
# Room definitions
# -----------------------
rooms = {
    "Living": {"min_area": 12, "max_area": 20},
    "Bedroom": {"min_area": 10, "max_area": 16},
    "Kitchen": {"min_area": 6, "max_area": 10},
    "Toilet": {"min_area": 3, "max_area": 5},
    "Dining": {"min_area": 8, "max_area": 12},
    "Balcony": {"min_area": 4, "max_area": 6}
}

preferred_neighbors = {
    "Living": ["Dining", "Kitchen", "Balcony"],
    "Bedroom": ["Toilet"],
    "Kitchen": ["Living", "Dining"],
    "Toilet": ["Bedroom"],
    "Dining": ["Living", "Kitchen"],
    "Balcony": ["Living"]
}

colors = {
    "Living": "skyblue",
    "Bedroom": "lightgreen",
    "Kitchen": "salmon",
    "Toilet": "navajowhite",
    "Dining": "violet",
    "Balcony": "khaki"
}

# -----------------------
# Helpers
# -----------------------
def area_to_wh(area):
    """Convert area (approx m^2) to integer-ish width & height (in grid units ~ meters)."""
    w = max(1, int(round(math.sqrt(area))))
    h = max(1, int(round(area / w)))
    return w, h

def clamp(v, a, b):
    return max(a, min(b, v))

def rects_overlap(a, b):
    return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or
                a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])

def shared_edge(a, b):
    """Return shared edge segment (x1,y1,x2,y2, orientation) if rectangles touch along an edge (not corner).
       Returns None if no shared straight wall.
    """
    # check vertical shared edge (a right == b left or a left == b right) with overlapping y-range
    eps = 1e-6
    # a right meets b left
    if abs((a["x"] + a["w"]) - b["x"]) < eps or abs((b["x"] + b["w"]) - a["x"]) < eps:
        # overlapping y-range?
        y1 = max(a["y"], b["y"])
        y2 = min(a["y"] + a["h"], b["y"] + b["h"])
        if y2 - y1 > 0.5:  # require some overlap
            # vertical shared edge segment
            if abs((a["x"] + a["w"]) - b["x"]) < eps:
                x = a["x"] + a["w"]
            else:
                x = b["x"] + b["w"]
            return (x, y1, x, y2, "V")
    # check horizontal shared edge (a bottom meets b top or vice versa)
    if abs((a["y"] + a["h"]) - b["y"]) < eps or abs((b["y"] + b["h"]) - a["y"]) < eps:
        x1 = max(a["x"], b["x"])
        x2 = min(a["x"] + a["w"], b["x"] + b["w"])
        if x2 - x1 > 0.5:
            if abs((a["y"] + a["h"]) - b["y"]) < eps:
                y = a["y"] + a["h"]
            else:
                y = b["y"] + b["h"]
            return (x1, y, x2, y, "H")
    return None

# -----------------------
# Layout generation within setbacks
# -----------------------
def place_with_adjacency(layout, name, grid_w, grid_h, w, h, setbacks):
    left, right, front, rear = setbacks
    build_x_min = left
    build_y_min = front
    build_w = grid_w - (left + right)
    build_h = grid_h - (front + rear)
    # try placing adjacent to preferred neighbors if exist
    candidates = []
    neighbors = preferred_neighbors.get(name, [])
    for n in neighbors:
        if n in layout:
            nx, ny, nw, nh = layout[n]["x"], layout[n]["y"], layout[n]["w"], layout[n]["h"]
            # attempt four adjacency positions
            options = [
                (nx + nw, ny),       # right of neighbor
                (nx - w, ny),        # left of neighbor
                (nx, ny + nh),       # below neighbor
                (nx, ny - h)         # above neighbor
            ]
            for (x, y) in options:
                if (build_x_min <= x <= build_x_min + build_w - w) and (build_y_min <= y <= build_y_min + build_h - h):
                    # quick overlap check
                    temp = {"x": x, "y": y, "w": w, "h": h}
                    if not any(rects_overlap(temp, r) for r in layout.values()):
                        candidates.append((x, y))
    if candidates:
        return random.choice(candidates)
    # fallback: try random positions inside buildable area avoiding overlaps
    tries = 0
    while tries < 40:
        x = random.randint(build_x_min, build_x_min + build_w - w)
        y = random.randint(build_y_min, build_y_min + build_h - h)
        temp = {"x": x, "y": y, "w": w, "h": h}
        if not any(rects_overlap(temp, r) for r in layout.values()):
            return x, y
        tries += 1
    # last resort: allow overlap but clamp
    return clamp(random.randint(build_x_min, build_x_min + build_w - w), build_x_min, build_x_min + build_w - w), \
           clamp(random.randint(build_y_min, build_y_min + build_h - h), build_y_min, build_y_min + build_h - h)

def random_room_layout(grid_w, grid_h, setbacks):
    left, right, front, rear = setbacks
    build_w = grid_w - (left + right)
    build_h = grid_h - (front + rear)
    layout = {}
    order = list(rooms.keys())
    random.shuffle(order)
    for name in order:
        area = random.randint(rooms[name]["min_area"], rooms[name]["max_area"])
        w, h = area_to_wh(area)
        # ensure fit
        w = min(w, build_w)
        h = min(h, build_h)
        x, y = place_with_adjacency(layout, name, grid_w, grid_h, w, h, setbacks)
        layout[name] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": area}
    return layout

# -----------------------
# Fitness (continuous, expressive)
# -----------------------
def fitness(layout, grid_w, grid_h, setbacks):
    score = 0.0
    # area compliance reward
    for name, r in layout.items():
        mn = rooms[name]["min_area"]
        mx = rooms[name]["max_area"]
        if mn <= r["area"] <= mx:
            score += 2.0
        else:
            score -= abs(r["area"] - (mn + mx) / 2) * 0.05
    # overlap heavy penalty (area of overlap)
    names = list(layout.keys())
    total_overlap = 0.0
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = layout[names[i]]
            b = layout[names[j]]
            if rects_overlap(a, b):
                ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
                iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
                total_overlap += ix * iy
    score -= total_overlap * 1.5
    # adjacency bonus (Manhattan/Euclidean mixed)
    for name, r in layout.items():
        for nb in preferred_neighbors.get(name, []):
            if nb in layout:
                rn = layout[nb]
                dx = abs((r["x"] + r["w"]/2) - (rn["x"] + rn["w"]/2))
                dy = abs((r["y"] + r["h"]/2) - (rn["y"] + rn["h"]/2))
                dist = math.hypot(dx, dy)
                score += max(0, 4.5 - dist) * 0.6
    # compactness: penalize spread
    xs = [r["x"] + r["w"]/2 for r in layout.values()]
    ys = [r["y"] + r["h"]/2 for r in layout.values()]
    spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
    score -= spread * 0.05
    # slight noise for tie-breaking (very small)
    score += random.uniform(-0.01, 0.01)
    return round(score, 4)

# -----------------------
# Mutation
# -----------------------
def mutate(layout, grid_w, grid_h, setbacks):
    new_layout = {k: dict(v) for k, v in layout.items()}
    left, right, front, rear = setbacks
    build_w = grid_w - (left + right)
    build_h = grid_h - (front + rear)
    for name, r in new_layout.items():
        if random.random() < 0.45:
            if random.random() < 0.25:
                # re-seed position
                r["x"] = random.randint(left, left + build_w - r["w"])
                r["y"] = random.randint(front, front + build_h - r["h"])
            else:
                r["x"] = clamp(r["x"] + random.randint(-2, 2), left, left + build_w - r["w"])
                r["y"] = clamp(r["y"] + random.randint(-2, 2), front, front + build_h - r["h"])
    return new_layout

# -----------------------
# GA
# -----------------------
def run_ga(grid_w, grid_h, setbacks, pop_size=50, generations=60, elitism=4):
    population = [random_room_layout(grid_w, grid_h, setbacks) for _ in range(pop_size)]
    for gen in range(generations):
        scored = sorted(population, key=lambda p: fitness(p, grid_w, grid_h, setbacks), reverse=True)
        next_gen = scored[:elitism]
        # breed from top 20
        top_pool = scored[:max(10, pop_size // 4)]
        while len(next_gen) < pop_size:
            parent = random.choice(top_pool)
            child = mutate(parent, grid_w, grid_h, setbacks)
            next_gen.append(child)
        population = next_gen
    # final refine: resolve overlaps greedily (sliding)
    final_best = max(population, key=lambda p: fitness(p, grid_w, grid_h, setbacks))
    return final_best

# -----------------------
# Doors & windows computation
# -----------------------
def compute_doors_windows(layout, grid_w, grid_h, setbacks):
    left, right, front, rear = setbacks
    build_x_min = left
    build_y_min = front
    build_x_max = grid_w - right
    build_y_max = grid_h - rear
    doors = []   # list of (x1,y1,x2,y2)
    windows = [] # list of (x1,y1,x2,y2)

    # doors between rooms where they share an edge
    names = list(layout.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = layout[names[i]]
            b = layout[names[j]]
            se = shared_edge(a, b)
            if se:
                x1, y1, x2, y2, orient = se
                # place a door in middle third of shared segment
                if orient == "V":
                    ymid = (y1 + y2) / 2
                    door_len = min(1.0, (y2 - y1) * 0.4)
                    doors.append((x1, ymid - door_len/2, x1, ymid + door_len/2))
                else:
                    xmid = (x1 + x2) / 2
                    door_len = min(1.0, (x2 - x1) * 0.4)
                    doors.append((xmid - door_len/2, y1, xmid + door_len/2, y1))

    # windows on external/buildable boundary walls
    for name, r in layout.items():
        # left edge touching buildable boundary?
        # touch tolerance eps
        eps = 0.001
        # left wall
        if abs(r["x"] - build_x_min) < eps:
            # vertical window on left wall
            y1 = r["y"] + max(0.2, 0.1 * r["h"])
            y2 = r["y"] + r["h"] - max(0.2, 0.1 * r["h"])
            if y2 - y1 > 0.3:
                windows.append((r["x"], y1, r["x"], y2))
        # right wall
        if abs((r["x"] + r["w"]) - build_x_max) < eps:
            y1 = r["y"] + max(0.2, 0.1 * r["h"])
            y2 = r["y"] + r["h"] - max(0.2, 0.1 * r["h"])
            if y2 - y1 > 0.3:
                windows.append((r["x"] + r["w"], y1, r["x"] + r["w"], y2))
        # top wall (y)
        if abs((r["y"] + r["h"]) - build_y_max) < eps:
            x1 = r["x"] + max(0.2, 0.1 * r["w"])
            x2 = r["x"] + r["w"] - max(0.2, 0.1 * r["w"])
            if x2 - x1 > 0.3:
                windows.append((x1, r["y"] + r["h"], x2, r["y"] + r["h"]))
        # bottom wall
        if abs(r["y"] - build_y_min) < eps:
            x1 = r["x"] + max(0.2, 0.1 * r["w"])
            x2 = r["x"] + r["w"] - max(0.2, 0.1 * r["w"])
            if x2 - x1 > 0.3:
                windows.append((x1, r["y"], x2, r["y"]))
    return doors, windows

# -----------------------
# Draw plan (matplotlib -> PIL)
# -----------------------
def draw_plan_image(layout, grid_w, grid_h, setbacks):
    doors, windows = compute_doors_windows(layout, grid_w, grid_h, setbacks)
    left, right, front, rear = setbacks
    build_x_min = left
    build_y_min = front
    build_x_max = grid_w - right
    build_y_max = grid_h - rear

    fig, ax = plt.subplots(figsize=(8, 10))
    # grid / site boundary
    ax.set_xlim(0, grid_w)
    ax.set_ylim(0, grid_h)
    # buildable area rectangle
    build_w = build_x_max - build_x_min
    build_h = build_y_max - build_y_min
    ax.add_patch(plt.Rectangle((build_x_min, build_y_min), build_w, build_h,
                 fill=False, edgecolor="red", linestyle="--", linewidth=1.8, label="Buildable Area"))

    # rooms
    for name, r in layout.items():
        rect = plt.Rectangle((r["x"], r["y"]), r["w"], r["h"], facecolor=colors.get(name, "lightgrey"),
                             edgecolor="black", alpha=0.9)
        ax.add_patch(rect)
        ax.text(r["x"] + r["w"]/2, r["y"] + r["h"]/2, f"{name}\n{r['area']}m²",
                ha="center", va="center", fontsize=9, weight="bold")

    # doors
    for (x1, y1, x2, y2) in doors:
        ax.plot([x1, x2], [y1, y2], color="brown", linewidth=4, solid_capstyle='butt')

    # windows
    for (x1, y1, x2, y2) in windows:
        ax.plot([x1, x2], [y1, y2], color="deepskyblue", linewidth=3, linestyle='-', solid_capstyle='butt')

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(f"Best Layout — Fitness: {fitness(layout, grid_w, grid_h, setbacks):.3f}")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)
    return img

# -----------------------
# Gradio wrapper
# -----------------------
def generate_best(site_width, site_height, front_sb, rear_sb, left_sb, right_sb):
    # cast to ints for grid
    grid_w = max(3, int(round(site_width)))
    grid_h = max(3, int(round(site_height)))
    # setbacks order: left, right, front, rear
    setbacks = (max(0, int(round(left_sb))),
               max(0, int(round(right_sb))),
               max(0, int(round(front_sb))),
               max(0, int(round(rear_sb))))
    # ensure buildable area positive
    if setbacks[0] + setbacks[1] >= grid_w or setbacks[2] + setbacks[3] >= grid_h:
        # invalid, return a blank image with message
        fig, ax = plt.subplots(figsize=(6,4))
        ax.text(0.5,0.5,"Invalid setbacks: buildable area non-positive", ha='center', va='center', fontsize=12)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)
        return img

    best_layout = run_ga(grid_w, grid_h, setbacks, pop_size=50, generations=60, elitism=5)
    img = draw_plan_image(best_layout, grid_w, grid_h, setbacks)
    return img

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## AI Space Planner — Best Intelligent Layout (with Setbacks, Doors & Windows)")
    with gr.Row():
        site_width = gr.Number(label="Site Width (m)", value=12, precision=0)
        site_height = gr.Number(label="Site Height (m)", value=15, precision=0)
    with gr.Row():
        front_sb = gr.Number(label="Front Setback (m)", value=1, precision=1)
        rear_sb = gr.Number(label="Rear Setback (m)", value=1, precision=1)
        left_sb = gr.Number(label="Left Setback (m)", value=1, precision=1)
        right_sb = gr.Number(label="Right Setback (m)", value=1, precision=1)
    btn = gr.Button("Generate Best Layout")
    out_img = gr.Image(type="pil")
    btn.click(generate_best, inputs=[site_width, site_height, front_sb, rear_sb, left_sb, right_sb], outputs=out_img)

if __name__ == "__main__":
    demo.launch() 