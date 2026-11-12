"""Synthetic office point cloud for testing cloud_to_floorplan.py."""
import numpy as np

rng = np.random.default_rng(0)
H = 2.8
T = 0.12  # wall thickness: lidar sees both faces

# walls in aligned frame: (x0,y0,x1,y1), door gaps carved out below
walls = [
    (0, 0, 24, 0), (24, 0, 24, 14), (24, 14, 0, 14), (0, 14, 0, 0),   # outer
    (0, 6, 24, 6), (0, 8, 24, 8),                                      # corridor
    (6, 0, 6, 6), (12, 0, 12, 6), (18, 0, 18, 6),                      # offices south
    (8, 8, 8, 14), (16, 8, 16, 14),                                    # offices north
]
doors = [(3, 6, 1.0), (9, 6, 1.0), (15, 6, 1.0), (21, 6, 1.0),
         (4, 8, 1.0), (12, 8, 1.0), (20, 8, 1.0)]


def sample_wall(x0, y0, x1, y1, n_per_m2=150):
    p0, p1 = np.array([x0, y0]), np.array([x1, y1])
    L = np.linalg.norm(p1 - p0)
    d = (p1 - p0) / L
    nrm = np.array([-d[1], d[0]])
    n = int(L * H * n_per_m2)
    t = rng.uniform(0, L, n)
    z = rng.uniform(0, H, n)
    side = rng.choice([-1, 1], n) * T / 2
    xy = p0 + t[:, None] * d + side[:, None] * nrm
    keep = np.ones(n, bool)
    for dx, dy, w in doors:            # carve door openings (2.1 m tall)
        c = np.array([dx, dy])
        along = (c - p0) @ d
        if abs((c - p0) @ nrm) < 0.01 and 0 <= along <= L:
            keep &= ~((np.abs(t - along) < w / 2) & (z < 2.1))
    # occlusion: randomly drop chunks of wall (missed by lidar)
    for _ in range(int(L / 4)):
        c0 = rng.uniform(0, L)
        ztop = H if rng.random() < 0.3 else 1.6        # mostly furniture in front of wall
        keep &= ~((t > c0) & (t < c0 + rng.uniform(0.2, 0.8)) & (z < ztop) & (rng.random(n) < 0.9))
    return np.c_[xy, z][keep]


pts = [sample_wall(*w) for w in walls]
# floor: sparse (lidar FOV), ceiling: denser
pts.append(np.c_[rng.uniform(0, 24, 8000), rng.uniform(0, 14, 8000), rng.normal(0, 0.01, 8000)])
pts.append(np.c_[rng.uniform(0, 24, 40000), rng.uniform(0, 14, 40000), H + rng.normal(0, 0.01, 40000)])
# furniture: desks (0.75 m) and cabinets
for _ in range(40):
    cx, cy = rng.uniform(1, 23), rng.uniform(1, 13)
    w, d, h = rng.uniform(0.6, 1.6), rng.uniform(0.5, 0.8), rng.choice([0.75, 1.2, 1.8])
    n = 1500
    pts.append(np.c_[cx + rng.uniform(0, w, n), cy + rng.uniform(0, d, n), rng.uniform(0, h, n)])
# ceiling clutter (lights, ducts) and random noise
pts.append(np.c_[rng.uniform(0, 24, 3000), rng.uniform(0, 14, 3000), rng.uniform(2.3, 2.8, 3000)])
pts.append(np.c_[rng.uniform(-1, 25, 2000), rng.uniform(-1, 15, 2000), rng.uniform(0, 3, 2000)])

P = np.vstack(pts)
P[:, :3] += rng.normal(0, 0.015, P.shape)
# put it in an arbitrary map frame: yaw 23 deg, offset, floor at z=-0.4
th = np.radians(23)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
P[:, :2] = P[:, :2] @ R.T + [5, -3]
P[:, 2] -= 0.4
np.save("synthetic_office.npy", P)
print(len(P), "points")
