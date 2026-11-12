#!/usr/bin/env python3
"""
cloud_to_floorplan.py -- Lidar point cloud (e.g. GLIM output) -> 2.5D office model
for ray tracing (Sionna RT / Mitsuba).

Pipeline
  1. Load + voxel downsample + outlier removal
  2. Floor / ceiling heights from z-histogram peaks
  3. Manhattan alignment: dominant wall direction from 2D normals in a wall slice
  4. Slice at wall height, rasterize to a 2D density image
  5. Extract axis-aligned wall segments (morphological line filtering)
  6. Merge two-sided walls, close gaps, reject furniture (must reach floor AND
     ceiling), snap corners, detect door openings
  7. Extrude walls floor->ceiling, add floor/ceiling slabs, rotate back to the
     original (GLIM) frame
  8. Export PLY meshes + Sionna RT scene XML + a debug PNG

Assumes z is (roughly) gravity-up, which is true for GLIM maps built with an IMU.
Assumes a Manhattan (rectilinear) building; for diagonal walls see the note in
extract_segments().

Usage
  python cloud_to_floorplan.py map.ply -o out_scene
  python cloud_to_floorplan.py map.ply -o out_scene --floor 0.0 --ceiling 2.8
"""
import argparse
import os

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 1. Loading and cleaning
# ---------------------------------------------------------------------------
def load_points(path):
    if path.endswith(".npy"):
        return np.load(path)[:, :3].astype(np.float64)
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def voxel_downsample(pts, voxel):
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    out = np.zeros((len(idx), 3))
    np.add.at(out, inv, pts)
    return out / np.bincount(inv)[:, None]


def remove_outliers(pts, k=16, std_ratio=2.0):
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    md = d[:, 1:].mean(1)
    return pts[md < md.mean() + std_ratio * md.std()]


# ---------------------------------------------------------------------------
# 2. Floor / ceiling
# ---------------------------------------------------------------------------
def floor_ceiling(z, bin_size=0.02):
    """Floor = strongest z-peak in the lower half, ceiling = strongest in the upper.
    Horizontal surfaces create sharp peaks; walls spread evenly over z."""
    bins = np.arange(z.min(), z.max() + bin_size, bin_size)
    h, e = np.histogram(z, bins)
    c = 0.5 * (e[1:] + e[:-1])
    peaks, _ = find_peaks(h, prominence=h.max() * 0.02)
    if len(peaks) == 0:
        raise RuntimeError("No horizontal surfaces found; pass --floor/--ceiling")
    mid = 0.5 * (z.min() + z.max())
    lo = [p for p in peaks if c[p] < mid]
    hi = [p for p in peaks if c[p] >= mid]
    floor = c[max(lo, key=lambda p: h[p])] if lo else np.percentile(z, 1)
    ceil = c[max(hi, key=lambda p: h[p])] if hi else np.percentile(z, 99)
    return float(floor), float(ceil)


# ---------------------------------------------------------------------------
# 3. Manhattan alignment
# ---------------------------------------------------------------------------
def dominant_yaw(xy, k=12):
    """Angle (rad, in [0, pi/2)) of the dominant wall direction.
    PCA on 2D neighbourhoods gives local line direction; fold mod 90 deg and
    take the peak of a weighted histogram."""
    _, nn = cKDTree(xy).query(xy, k=k)
    nb = xy[nn] - xy[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    w, v = np.linalg.eigh(cov)
    d = v[:, :, 1]                                  # principal direction
    lin = 1.0 - w[:, 0] / np.maximum(w[:, 1], 1e-12)  # linearity weight
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi / 2)
    nb_ = 180
    h, e = np.histogram(ang, nb_, (0, np.pi / 2), weights=lin)
    h = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5) / 5, "valid")  # circular smooth
    return 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
# 4-5. Density image and segment extraction
# ---------------------------------------------------------------------------
def density_image(xy, res):
    origin = xy.min(0) - 5 * res
    ij = np.floor((xy - origin) / res).astype(int)
    shape = ij.max(0) + 6
    img = np.zeros((shape[1], shape[0]), np.float32)  # rows = y, cols = x
    np.add.at(img, (ij[:, 1], ij[:, 0]), 1)
    return img, origin


def extract_segments(occ, res, origin, min_len, gap):
    """Axis-aligned wall segments from a binary occupancy image.

    Opening with a long thin kernel keeps only structures that are straight
    and at least `min_len` long in that direction (furniture blobs vanish).
    Each connected component becomes one segment.

    Non-Manhattan buildings: replace this with cv2.HoughLinesP on `occ`
    (or LSD) and skip the snapping step.
    """
    L = max(3, int(min_len / res))
    G = max(1, int(gap / res))
    segs = []  # (axis, offset, start, end, support)  axis 0: along x, 1: along y
    for axis in (0, 1):
        kclose = cv2.getStructuringElement(cv2.MORPH_RECT, (G, 1) if axis == 0 else (1, G))
        kopen = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if axis == 0 else (1, L))
        m = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kclose)   # bridge small gaps along wall
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kopen)       # keep long straight runs
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for c in range(1, n):
            rr, cc = np.nonzero(lab == c)
            if axis == 0:
                off, a, b = rr.mean(), cc.min(), cc.max() + 1
                off_w, s0, s1 = origin[1] + (off + 0.5) * res, origin[0] + a * res, origin[0] + b * res
            else:
                off, a, b = cc.mean(), rr.min(), rr.max() + 1
                off_w, s0, s1 = origin[0] + (off + 0.5) * res, origin[1] + a * res, origin[1] + b * res
            segs.append([axis, off_w, s0, s1, len(rr)])
    return segs


# ---------------------------------------------------------------------------
# 6. Merge, close gaps, snap corners
# ---------------------------------------------------------------------------
def merge_parallel(segs, max_sep, max_gap):
    """Merge segments of the same axis whose offsets are within max_sep and
    whose spans overlap or are within max_gap. Two faces of one wall -> one wall."""
    out = []
    for axis in (0, 1):
        S = sorted([s for s in segs if s[0] == axis], key=lambda s: s[1])
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a, b = S[i], S[j]
                    if abs(a[1] - b[1]) <= max_sep and \
                       min(a[3], b[3]) - max(a[2], b[2]) >= -max_gap:
                        w = a[4] + b[4]
                        S[i] = [axis, (a[1] * a[4] + b[1] * b[4]) / w,
                                min(a[2], b[2]), max(a[3], b[3]), w]
                        S.pop(j)
                        changed = True
                        break
                if changed:
                    break
        out += S
    return out


def snap_corners(segs, tol):
    """Extend/trim each endpoint onto a perpendicular wall if it is within tol."""
    for s in segs:
        perp = [p for p in segs if p[0] != s[0]]
        for k in (2, 3):
            best, bd = None, tol
            for p in perp:
                # p's offset is a coordinate along s; s's offset must lie in p's span
                if p[2] - tol <= s[1] <= p[3] + tol:
                    d = abs(p[1] - s[k])
                    if d < bd:
                        best, bd = p[1], d
            if best is not None:
                s[k] = best
    return [s for s in segs if s[3] - s[2] > 1e-3]


# ---------------------------------------------------------------------------
# 6b. Use full-height evidence: reject furniture, find door openings
# ---------------------------------------------------------------------------
def points_near(seg, P, halfw):
    """Aligned-frame points within halfw of the segment line and inside its span.
    Returns (coordinate along wall, z)."""
    axis, off, a, b = seg[:4]
    u, v = (P[:, 0], P[:, 1]) if axis == 0 else (P[:, 1], P[:, 0])
    m = (np.abs(v - off) < halfw) & (u >= a) & (u <= b)
    return u[m], P[m, 2]


def is_full_height(seg, P, fz, cz, halfw, min_frac=0.4, min_pts=5):
    """A real wall has points both low (near floor) and high (near ceiling)
    along a good fraction of its length. Desks/cabinets fail the high test,
    ceiling ducts/lights fail the low test."""
    u, z = points_near(seg, P, halfw)
    a, b = seg[2], seg[3]
    bins = np.arange(a, b + 0.25, 0.25)
    def frac(zlo, zhi):
        m = (z > zlo) & (z < zhi)
        return np.mean(np.histogram(u[m], bins)[0] >= min_pts) if len(bins) > 1 else 0
    return frac(fz + 0.2, fz + 0.8) >= min_frac and frac(cz - 0.6, cz - 0.1) >= min_frac


def find_doors(seg, P, fz, cz, halfw, door_w=(0.6, 2.0), bin_=0.05, min_pts=3):
    """Door = a run along the wall with no points in the 0.3-1.9 m band.
    Closed doors return points, so they correctly stay solid.
    Returns list of (start, end, top_z)."""
    u, z = points_near(seg, P, halfw)
    a, b = seg[2], seg[3]
    bins = np.arange(a, b + bin_, bin_)
    if len(bins) < 3:
        return []
    # a few stray points in a doorway should not close it
    low = np.histogram(u[(z > fz + 0.3) & (z < fz + 1.9)], bins)[0] >= min_pts
    doors, i = [], 0
    while i < len(low):
        if not low[i]:
            j = i
            while j < len(low) and not low[j]:
                j += 1
            s0, s1 = bins[i], bins[j]
            if door_w[0] <= s1 - s0 <= door_w[1] and i > 0 and j < len(low):
                above = z[(u > s0) & (u < s1) & (z > fz + 1.9)]
                top = float(np.percentile(above, 2)) if len(above) > 10 else fz + 2.1
                doors.append((s0, s1, min(top, cz - 0.05)))
            i = j
        else:
            i += 1
    return doors


# ---------------------------------------------------------------------------
# 7. Meshing
# ---------------------------------------------------------------------------
def seg_to_xy(s):
    axis, off, a, b = s[:4]
    return (np.array([[a, off], [b, off]]) if axis == 0
            else np.array([[off, a], [off, b]]))


def wall_mesh(p0, p1, z0, z1, thickness=0.0):
    """Quad (thickness=0) or box for one wall, endpoints in world xy."""
    if thickness <= 0:
        V = np.array([[*p0, z0], [*p1, z0], [*p1, z1], [*p0, z1]])
        F = np.array([[0, 1, 2], [0, 2, 3]])
        return V, F
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * thickness / 2
    q = [p0 - n, p1 - n, p1 + n, p0 + n]
    V = np.array([[*p, z] for z in (z0, z1) for p in q])
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def wall_pieces(seg, doors, fz, cz, Rinv, thickness):
    """Mesh one wall, leaving door openings (with lintel above each door)."""
    axis, off, a, b = seg[:4]
    def to_world(s0, s1):
        p = seg_to_xy([axis, off, s0, s1]) @ Rinv.T
        return p[0], p[1]
    out, cur = [], a
    for d0, d1, top in sorted(doors):
        if d0 > cur:
            out.append(wall_mesh(*to_world(cur, d0), fz, cz, thickness))
        out.append(wall_mesh(*to_world(d0, d1), top, cz, thickness))  # lintel
        cur = d1
    if b > cur:
        out.append(wall_mesh(*to_world(cur, b), fz, cz, thickness))
    return out


def concat(meshes):
    Vs, Fs, o = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + o)
        o += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def write_ply(path, V, F):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in V:
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for t in F:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def write_sionna_xml(path, objects):
    """objects: list of (name, ply_rel_path, itu_type, thickness)."""
    mats = {(t, th) for _, _, t, th in objects}
    lines = ['<scene version="2.1.0">']
    for t, th in sorted(mats):
        lines += [f'  <bsdf type="itu-radio-material" id="mat-{t}-{th:g}">',
                  f'    <string name="type" value="{t}"/>',
                  f'    <float name="thickness" value="{th}"/>',
                  '  </bsdf>']
    for name, ply, t, th in objects:
        lines += [f'  <shape type="ply" id="mesh-{name}">',
                  f'    <string name="filename" value="{ply}"/>',
                  '    <boolean name="face_normals" value="true"/>',
                  f'    <ref id="mat-{t}-{th:g}" name="bsdf"/>',
                  '  </shape>']
    lines.append("</scene>")
    open(path, "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cloud", help=".ply/.pcd/.npy point cloud")
    ap.add_argument("-o", "--out", default="scene_out")
    ap.add_argument("--voxel", type=float, default=0.03, help="downsample voxel [m]")
    ap.add_argument("--res", type=float, default=0.04, help="floor-plan pixel size [m]")
    ap.add_argument("--floor", type=float, help="override floor z")
    ap.add_argument("--ceiling", type=float, help="override ceiling z")
    ap.add_argument("--slice", type=float, nargs=2, default=[0.9, -0.25],
                    help="wall slice [m]: start above floor, end relative to ceiling (negative = below "
                         "ceiling). Going up past door height lets lintels bridge doorways.")
    ap.add_argument("--min-hits", type=float, default=2, help="points per pixel to count as occupied")
    ap.add_argument("--min-len", type=float, default=0.6, help="shortest wall to keep [m]")
    ap.add_argument("--gap", type=float, default=0.5, help="bridge holes in a wall up to this [m]")
    ap.add_argument("--merge-sep", type=float, default=0.30, help="merge parallel faces closer than this [m]")
    ap.add_argument("--snap", type=float, default=0.5, help="corner snapping tolerance [m]")
    ap.add_argument("--wall-thickness", type=float, default=0.0,
                    help="0 = thin quads (thickness only in material), >0 = boxes")
    ap.add_argument("--no-doors", action="store_true", help="do not carve door openings")
    ap.add_argument("--keep-partial", action="store_true",
                    help="keep segments that do not reach floor and ceiling (cubicles, cabinets)")
    ap.add_argument("--wall-material", default="plasterboard")
    ap.add_argument("--wall-material-thickness", type=float, default=0.12)
    a = ap.parse_args(argv)

    os.makedirs(os.path.join(a.out, "meshes"), exist_ok=True)

    pts = load_points(a.cloud)
    print(f"loaded {len(pts):,} points")
    pts = remove_outliers(voxel_downsample(pts, a.voxel))
    print(f"after downsample/outliers: {len(pts):,}")

    fz, cz = floor_ceiling(pts[:, 2])
    fz = a.floor if a.floor is not None else fz
    cz = a.ceiling if a.ceiling is not None else cz
    print(f"floor z = {fz:.3f}, ceiling z = {cz:.3f}  (height {cz - fz:.2f} m)")

    sl = pts[(pts[:, 2] > fz + a.slice[0]) & (pts[:, 2] < cz + a.slice[1])]
    yaw = dominant_yaw(sl[:, :2])
    print(f"dominant wall direction: {np.degrees(yaw):.2f} deg")
    R = rot2(-yaw)                       # world -> aligned
    xy = sl[:, :2] @ R.T

    img, origin = density_image(xy, a.res)
    occ = (img >= a.min_hits).astype(np.uint8) * 255

    segs = extract_segments(occ, a.res, origin, a.min_len, a.gap)
    print(f"raw segments: {len(segs)}")
    segs = merge_parallel(segs, a.merge_sep, a.gap)
    P_al = np.c_[pts[:, :2] @ R.T, pts[:, 2]]          # whole cloud, aligned frame
    halfw = a.merge_sep / 2 + 0.05
    if not a.keep_partial:
        n0 = len(segs)
        segs = [s for s in segs if is_full_height(s, P_al, fz, cz, halfw)]
        print(f"rejected {n0 - len(segs)} partial-height segments (furniture/ducts)")
    segs = snap_corners(segs, a.snap)
    segs = [s for s in segs if s[3] - s[2] >= a.min_len]
    doors = [[] if a.no_doors else find_doors(s, P_al, fz, cz, halfw) for s in segs]
    print(f"final walls: {len(segs)}, door openings: {sum(map(len, doors))}")

    # --- meshes in world frame
    Rinv = R.T
    walls = []
    for s, d in zip(segs, doors):
        walls += wall_pieces(s, d, fz, cz, Rinv, a.wall_thickness)
    Vw, Fw = concat(walls)

    # floor/ceiling: rectangle covering the walls (aligned frame), rotated back
    allxy = np.vstack([seg_to_xy(s) for s in segs])
    lo, hi = allxy.min(0) - 0.05, allxy.max(0) + 0.05
    rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) @ Rinv.T
    Vf = np.c_[rect, np.full(4, fz)]
    Vc = np.c_[rect, np.full(4, cz)]
    Fq = np.array([[0, 1, 2], [0, 2, 3]])

    write_ply(os.path.join(a.out, "meshes", "walls.ply"), Vw, Fw)
    write_ply(os.path.join(a.out, "meshes", "floor.ply"), Vf, Fq)
    write_ply(os.path.join(a.out, "meshes", "ceiling.ply"), Vc, Fq[:, ::-1])
    write_sionna_xml(os.path.join(a.out, "scene.xml"), [
        ("walls", "meshes/walls.ply", a.wall_material, a.wall_material_thickness),
        ("floor", "meshes/floor.ply", "concrete", 0.2),
        ("ceiling", "meshes/ceiling.ply", "ceiling_board", 0.02),
    ])

    # --- debug image: density + occupancy + walls (aligned frame)
    vis = cv2.cvtColor(np.clip(img * 40, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis[occ > 0] = (90, 90, 90)
    for s, dd in zip(segs, doors):
        p = ((seg_to_xy(s) - origin) / a.res).astype(int)
        cv2.line(vis, tuple(p[0]), tuple(p[1]), (0, 0, 255), 2)
        for d0, d1, _ in dd:
            q = ((seg_to_xy([s[0], s[1], d0, d1]) - origin) / a.res).astype(int)
            cv2.line(vis, tuple(q[0]), tuple(q[1]), (0, 255, 0), 3)
    cv2.imwrite(os.path.join(a.out, "floorplan_debug.png"), cv2.flip(vis, 0))

    with open(os.path.join(a.out, "walls.csv"), "w") as f:
        f.write("x0,y0,x1,y1\n")
        for s in segs:
            p = seg_to_xy(s) @ Rinv.T
            f.write(f"{p[0,0]:.3f},{p[0,1]:.3f},{p[1,0]:.3f},{p[1,1]:.3f}\n")
    print(f"wrote {a.out}/scene.xml, meshes/, walls.csv, floorplan_debug.png")


if __name__ == "__main__":
    main()
