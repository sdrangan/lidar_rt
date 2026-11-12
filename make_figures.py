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
furn = c.extract_furniture(P_al, final, fz, cz, halfw)
KIND_COL = {"table": (0.95, 0.65, 0.2, 1.0), "object": (0.75, 0.3, 0.75, 1.0), "tall": (0.45, 0.45, 0.5, 1.0)}
KIND_LBL = {"table": "desk/table top (thin slab)", "object": "chair / low object (box)", "tall": "cabinet / shelf (box)"}

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
def fig_model(ax, with_furn=False, title="6. Extruded model (ceiling hidden)"):
    polys, cols = [], []
    for s, dd in zip(final, doors):
        for V, F in c.wall_pieces(s, dd, fz, cz, np.eye(2), 0.0):
            polys.append(V)
            cols.append((0.35, 0.6, 0.8, 0.85))
    lo = np.min([p[:, :2].min(0) for p in polys], 0)
    hi = np.max([p[:, :2].max(0) for p in polys], 0)
    polys.append(np.array([[lo[0], lo[1], fz], [hi[0], lo[1], fz], [hi[0], hi[1], fz], [lo[0], hi[1], fz]]))
    cols.append((0.75, 0.75, 0.75, 1.0))
    if with_furn:
        for f in furn:
            V, F = c.box_mesh(f["corners"], f["z0"], f["z1"])
            for t in F:
                polys.append(V[t])
                cols.append(KIND_COL[f["kind"]])
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=(0.1, 0.2, 0.3, 0.6), linewidths=0.3))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(fz, cz)
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], (cz - fz) * 2), zoom=1.05)
    ax.view_init(elev=50, azim=-65)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def fig_furniture(ax):
    draw_segs(ax, final, color="tab:red", lw=2)
    for f in furn:
        p = np.vstack([f["corners"], f["corners"][:1]])
        ax.fill(p[:, 0], p[:, 1], color=KIND_COL[f["kind"]], alpha=0.6)
        ax.plot(p[:, 0], p[:, 1], color=KIND_COL[f["kind"]][:3], lw=1)
    for k in KIND_COL:
        n = sum(f["kind"] == k for f in furn)
        ax.fill([], [], color=KIND_COL[k], alpha=0.6, label=f"{KIND_LBL[k]}: {n}")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plan_axes(ax, "Optional: furniture from leftover points (--furniture)")


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
plt.close(fig)

# ---- furniture option: plan and with/without comparison
fig = plt.figure(figsize=(7, 4.6))
fig_furniture(fig.add_subplot(111))
fig.savefig(os.path.join(out, "7_furniture_plan.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(15, 6))
fig_model(fig.add_subplot(121, projection="3d"), False, "Without furniture (default)")
fig_model(fig.add_subplot(122, projection="3d"), True, "With --furniture")
fig.tight_layout()
fig.savefig(os.path.join(out, "8_model_compare.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("wrote figures to", out)


# ---- radio maps with / without furniture (needs sionna-rt)
def radio_maps():
    import subprocess
    import mitsuba as mi
    from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver
    here = os.path.dirname(os.path.abspath(__file__))
    gains = {}
    for tag, extra in [("without", []), ("with", ["--furniture"])]:
        d = os.path.join("/tmp", f"lidar_rt_fig_{tag}")
        subprocess.run([sys.executable, os.path.join(here, "cloud_to_floorplan.py"), src, "-o", d] + extra,
                       check=True, capture_output=True)
        scene = load_scene(os.path.join(d, "scene.xml"))
        scene.frequency = 28e9
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        # AP near the ceiling in the middle of the corridor (aligned frame -> map frame)
        # corridor = between the two longest interior walls along x
        hz = [s_ for s_ in final if s_[0] == 0]
        ymin, ymax = min(s_[1] for s_ in hz), max(s_[1] for s_ in hz)
        inner = sorted([s_ for s_ in hz if ymin + 0.5 < s_[1] < ymax - 0.5], key=lambda s_: s_[2] - s_[3])[:2]
        mid_y = np.mean([s_[1] for s_ in inner])
        mid_x = 0.5 * (min(s_[2] for s_ in inner) + max(s_[3] for s_ in inner))
        txy = np.array([mid_x, mid_y]) @ R          # aligned -> world (R^-1 = R^T)
        scene.add(Transmitter("ap", position=[float(txy[0]), float(txy[1]), fz + 2.3]))
        allxy = np.vstack([c.seg_to_xy(s_) for s_ in final])
        lo, hi = allxy.min(0), allxy.max(0)
        ctr = 0.5 * (lo + hi) @ R
        rm = RadioMapSolver()(scene, center=mi.Point3f(float(ctr[0]), float(ctr[1]), float(fz + 1.0)),
                              orientation=mi.Point3f(float(yaw), 0.0, 0.0), size=mi.Point2f(float(hi[0] - lo[0]), float(hi[1] - lo[1])),
                              cell_size=mi.Point2f(0.1, 0.1), samples_per_tx=4 * 10**6, max_depth=5)
        gains[tag] = 10 * np.log10(np.maximum(rm.path_gain.numpy()[0], 1e-20))
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, tag in zip(axs, ["without", "with"]):
        im = ax.imshow(gains[tag], origin="lower", vmin=-120, vmax=-50, cmap="viridis")
        ax.set_title(f"28 GHz path gain at 1 m, {tag} furniture", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="dB", fraction=0.03)
    diff = np.clip(gains["with"], -150, None) - np.clip(gains["without"], -150, None)
    im = axs[2].imshow(diff, origin="lower", vmin=-20, vmax=20, cmap="RdBu")
    axs[2].set_title("Difference (with - without)", fontsize=11)
    axs[2].axis("off")
    fig.colorbar(im, ax=axs[2], label="dB", fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "9_radio_compare.png"), dpi=DPI, bbox_inches="tight")
    print("wrote radio map comparison")


try:
    radio_maps()
except ImportError:
    print("sionna-rt not installed; skipping radio map comparison")
