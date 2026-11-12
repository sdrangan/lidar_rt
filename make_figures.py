"""Generate the pipeline figures in docs/ from a point cloud (default: synthetic office).

Usage:
    python make_synthetic_office.py
    python make_figures.py synthetic_office.npy docs
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import cloud_to_floorplan as c

src = sys.argv[1] if len(sys.argv) > 1 else "synthetic_office.npy"
out = sys.argv[2] if len(sys.argv) > 2 else "docs"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)

# ---- run the pipeline step by step (same defaults as cloud_to_floorplan.py)
res, min_len, gap, sep, snap = 0.04, 0.6, 0.5, 0.30, 0.5
raw = c.load_points(src)
pts = c.remove_outliers(c.voxel_downsample(raw, 0.03))
fz, cz = c.floor_ceiling(pts[:, 2])
sl = pts[(pts[:, 2] > fz + 0.9) & (pts[:, 2] < cz - 0.25)]
yaw = c.dominant_yaw(sl[:, :2])
R = c.rot2(-yaw)
xy = sl[:, :2] @ R.T
img, origin = c.density_image(xy, res)
occ = (img >= 2).astype(np.uint8) * 255
raw_segs = c.extract_segments(occ, res, origin, min_len, gap)
merged = c.merge_parallel([list(s) for s in raw_segs], sep, gap)
P_al = np.c_[pts[:, :2] @ R.T, pts[:, 2]]
halfw = sep / 2 + 0.05
keep = [s for s in merged if c.is_full_height(s, P_al, fz, cz, halfw)]
rejected = [s for s in merged if not c.is_full_height(s, P_al, fz, cz, halfw)]
final = c.snap_corners([list(s) for s in keep], snap)
final = [s for s in final if s[3] - s[2] >= min_len]
doors = [c.find_doors(s, P_al, fz, cz, halfw) for s in final]

ext = [origin[0], origin[0] + img.shape[1] * res, origin[1], origin[1] + img.shape[0] * res]
DPI = 120


def plan_axes(ax, title):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def draw_segs(ax, segs, **kw):
    for s in segs:
        p = c.seg_to_xy(s)
        ax.plot(p[:, 0], p[:, 1], **kw)


# ---- 1. input cloud (3D)
def fig_input(ax):
    s = pts[rng.choice(len(pts), min(40000, len(pts)), replace=False)]
    s = s[s[:, 2] < cz - 0.1]                      # hide ceiling so the inside is visible
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s[:, 2], s=0.3, cmap="turbo")
    ax.set_title("1. Input point cloud (ceiling hidden)", fontsize=11)
    ax.view_init(elev=55, azim=-60)
    ax.set_xlim(s[:, 0].min(), s[:, 0].max()); ax.set_ylim(s[:, 1].min(), s[:, 1].max())
    ax.set_box_aspect((np.ptp(s[:, 0]), np.ptp(s[:, 1]), np.ptp(s[:, 2]) * 2), zoom=1.5)
    ax.set_axis_off()


# ---- 2. height histogram
def fig_hist(ax):
    ax.hist(pts[:, 2], bins=200, color="0.4")
    ax.axvline(fz, color="tab:blue", lw=2, label=f"floor {fz:.2f} m")
    ax.axvline(cz, color="tab:red", lw=2, label=f"ceiling {cz:.2f} m")
    ax.axvspan(fz + 0.9, cz - 0.25, color="tab:orange", alpha=0.15, label="wall slice")
    ax.set_yscale("log")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("points")
    ax.set_title("2. Floor/ceiling from z-histogram peaks", fontsize=11)
    ax.legend(fontsize=8)


# ---- 3. density image of the slice (aligned frame)
def fig_density(ax):
    ax.imshow(np.log1p(img), origin="lower", extent=ext, cmap="magma")
    plan_axes(ax, f"3. Wall slice density (rotated {np.degrees(yaw):.1f}°)")


# ---- 4. line filtering
def fig_lines(ax):
    ax.imshow(occ, origin="lower", extent=ext, cmap="gray_r", alpha=0.35)
    draw_segs(ax, raw_segs, color="tab:red", lw=1)
    plan_axes(ax, f"4. Long straight runs: {len(raw_segs)} raw segments")


# ---- 5. merge, reject, doors
def fig_final(ax):
    ax.imshow(occ, origin="lower", extent=ext, cmap="gray_r", alpha=0.25)
    draw_segs(ax, rejected, color="0.6", lw=2, ls=":")
    draw_segs(ax, final, color="tab:red", lw=2)
    for s, dd in zip(final, doors):
        for d0, d1, _ in dd:
            p = c.seg_to_xy([s[0], s[1], d0, d1])
            ax.plot(p[:, 0], p[:, 1], color="tab:green", lw=5, solid_capstyle="butt")
    ax.plot([], [], color="tab:red", lw=2, label=f"walls ({len(final)})")
    ax.plot([], [], color="0.6", lw=2, ls=":", label=f"rejected: not full height ({len(rejected)})")
    ax.plot([], [], color="tab:green", lw=5, label=f"doors ({sum(map(len, doors))})")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plan_axes(ax, "5. Merged walls, furniture rejected, doors")


# ---- 6. extruded 3D model
def fig_model(ax):
    polys, cols = [], []
    for s, dd in zip(final, doors):
        for V, F in c.wall_pieces(s, dd, fz, cz, np.eye(2), 0.0):
            polys.append(V)
            cols.append((0.35, 0.6, 0.8, 0.85))
    lo = np.min([p[:, :2].min(0) for p in polys], 0)
    hi = np.max([p[:, :2].max(0) for p in polys], 0)
    polys.append(np.array([[lo[0], lo[1], fz], [hi[0], lo[1], fz], [hi[0], hi[1], fz], [lo[0], hi[1], fz]]))
    cols.append((0.75, 0.75, 0.75, 1.0))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=(0.1, 0.2, 0.3, 0.6), linewidths=0.3))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(fz, cz)
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], (cz - fz) * 2), zoom=1.05)
    ax.view_init(elev=50, azim=-65)
    ax.set_title("6. Extruded model (ceiling hidden)", fontsize=11)
    ax.set_axis_off()


panels = [("1_input.png", fig_input, True), ("2_height_hist.png", fig_hist, False),
          ("3_slice_density.png", fig_density, False), ("4_line_filter.png", fig_lines, False),
          ("5_walls_doors.png", fig_final, False), ("6_model_3d.png", fig_model, True)]

for name, fn, is3d in panels:
    fig = plt.figure(figsize=(7, 4.6))
    ax = fig.add_subplot(111, projection="3d" if is3d else None)
    fn(ax)
    fig.savefig(os.path.join(out, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

fig = plt.figure(figsize=(18, 10))
for i, (name, fn, is3d) in enumerate(panels):
    ax = fig.add_subplot(2, 3, i + 1, projection="3d" if is3d else None)
    fn(ax)
fig.tight_layout()
fig.savefig(os.path.join(out, "pipeline.png"), dpi=90, bbox_inches="tight")
print("wrote figures to", out)
