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
# ---- furniture (lidar sees surfaces, so sample surfaces, not volumes)
def box_surface(cx, cy, w, d, z0, z1, yaw=0.0, n_per_m2=400):
    """Points on the 4 sides + top of a box centred at (cx, cy)."""
    faces = []
    for (a, b, h) in [(w, d, None), (w, z1 - z0, "y"), (d, z1 - z0, "x")]:
        n = max(20, int(a * b * n_per_m2))
        if h is None:                                   # top
            faces.append(np.c_[rng.uniform(-w/2, w/2, n), rng.uniform(-d/2, d/2, n), np.full(n, z1)])
        elif h == "y":                                  # front/back
            s = rng.choice([-1, 1], n) * d / 2
            faces.append(np.c_[rng.uniform(-w/2, w/2, n), s, rng.uniform(z0, z1, n)])
        else:                                           # left/right
            s = rng.choice([-1, 1], n) * w / 2
            faces.append(np.c_[s, rng.uniform(-d/2, d/2, n), rng.uniform(z0, z1, n)])
    F = np.vstack(faces)
    c, s_ = np.cos(yaw), np.sin(yaw)
    F[:, :2] = F[:, :2] @ np.array([[c, -s_], [s_, c]]).T + [cx, cy]
    return F


def clear_of_walls(cx, cy, r):
    for x0, y0, x1, y1 in walls:
        if x0 == x1 and min(y0, y1) - r < cy < max(y0, y1) + r and abs(cx - x0) < r:
            return False
        if y0 == y1 and min(x0, x1) - r < cx < max(x0, x1) + r and abs(cy - y0) < r:
            return False
    return 0.3 < cx < 23.7 and 0.3 < cy < 13.7


def place(r):
    while True:
        cx, cy = rng.uniform(0, 24), rng.uniform(0, 14)
        if clear_of_walls(cx, cy, r):
            return cx, cy


truth = {"desk": 0, "chair": 0, "cabinet": 0}
for _ in range(14):                       # desks: thin top + 4 legs, hollow underneath
    w, d = rng.uniform(1.2, 1.6), rng.uniform(0.6, 0.8)
    cx, cy = place(np.hypot(w, d) / 2 + 0.2)
    pts.append(box_surface(cx, cy, w, d, 0.72, 0.75))
    for sx in (-1, 1):
        for sy in (-1, 1):
            pts.append(box_surface(cx + sx * (w/2 - 0.05), cy + sy * (d/2 - 0.05), 0.04, 0.04, 0, 0.72))
    truth["desk"] += 1
for _ in range(14):                       # chairs: seat + back, random yaw
    cx, cy = place(0.5)
    yaw = rng.uniform(0, np.pi)
    pts.append(box_surface(cx, cy, 0.45, 0.45, 0.40, 0.48, yaw))
    bx, by = cx + 0.22 * np.sin(yaw), cy - 0.22 * np.cos(yaw)   # back edge of seat
    pts.append(box_surface(bx, by, 0.45, 0.05, 0.48, 0.95, yaw))
    pts.append(box_surface(cx, cy, 0.06, 0.06, 0.0, 0.40, yaw))
    truth["chair"] += 1
for _ in range(8):                        # filing cabinets / bookshelves
    w, d, h = rng.uniform(0.5, 1.2), rng.uniform(0.4, 0.6), rng.choice([1.3, 1.8])
    cx, cy = place(np.hypot(w, d) / 2 + 0.2)
    pts.append(box_surface(cx, cy, w, d, 0.0, h))
    truth["cabinet"] += 1
print("ground truth furniture:", truth)
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
