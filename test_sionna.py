"""Smoke test: load a scene.xml from cloud_to_floorplan.py in Sionna RT, trace paths
and compute a radio map. Tx/Rx positions are set for the synthetic office;
edit w() / positions for a real map.

Usage: python test_sionna.py test_scene/scene.xml [radio_map.png]
"""
import sys, numpy as np
import mitsuba as mi
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver, RadioMapSolver
scene = load_scene(sys.argv[1])
for name, obj in scene.objects.items():
    print(name, obj.radio_material.name)
scene.frequency = 3.5e9
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = scene.tx_array
# put tx in corridor, rx inside an office (synthetic frame -> map frame)
th=np.radians(23); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
def w(x,y): p=R@[x,y]+[5,-3]; return [float(p[0]),float(p[1]),1.2]
scene.add(Transmitter("tx", position=w(12,7)))
scene.add(Receiver("rx", position=w(3,3)))
paths = PathSolver()(scene, max_depth=4, refraction=True)
a, tau = paths.cir(out_type="numpy")
print("paths:", np.count_nonzero(np.abs(a)), " path gain dB: %.1f" % (10*np.log10(np.sum(np.abs(a)**2))))
rm = RadioMapSolver()(scene, cell_size=(0.25,0.25), samples_per_tx=10**6, max_depth=3)
print("radio map shape:", rm.path_gain.numpy().shape)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
g = 10*np.log10(np.maximum(rm.path_gain.numpy()[0], 1e-20))
plt.figure(figsize=(8,5)); plt.imshow(g, origin="lower", vmin=-110, vmax=-40, cmap="viridis")
plt.colorbar(label="path gain [dB]"); plt.title("Sionna RT radio map, 3.5 GHz, synthetic office"); plt.axis("off")
plt.savefig(sys.argv[2] if len(sys.argv) > 2 else "radio_map_test.png", dpi=110, bbox_inches="tight")
