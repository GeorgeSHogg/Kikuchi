import kikuchipy as kp
import numpy as np
import orix

from fastai.vision.all import *
from PIL import Image
from orix.quaternion import Rotation, Symmetry

steel_types = ["a", "f"] #, "c", "s"]
austenite = kp.load("steel_h5s/a.h5")
detector = kp.detectors.EBSDDetector(
        shape=austenite.axes_manager.signal_shape[::-1],
        sample_tilt=70,
        pc=[0.421, 0.7794, 0.5049],
        convention="edax",
    )

def get_mps(steel_type):
    mp = kp.load("steel_h5s/" + steel_type + ".h5")
    return mp, mp.as_lambert()

def generate_batch(steel_type, mp, mp_lp, num = 100):
    batch_size = 100
    batches = int(num / batch_size)
    for i in range(batches):
        print("Starting", steel_type, "batch", str(i + 1), "/", batches)
        grains = np.deg2rad(np.random.rand(batch_size, 3) * 720 - 360)
        
        rot = Rotation.from_euler(grains)
        full_pat = mp_lp.get_patterns(rot, detector, energy=20, compute=True)

        sym = mp.phase.point_group
        refrence = torch.Tensor([[1, 0, 0, 0]])

        for i in range(batch_size):
            target_sym= Symmetry(rot[i])

            all_targets = orix.quaternion.symmetry.get_distinguished_points(sym, target_sym)
            target = torch.Tensor(all_targets.data)

            dots = torch.sum(target * refrence, 1, True)
            theta = torch.acos(2 * torch.square(dots) - 1)
            closest_theta = torch.min(theta)
            locs = torch.nonzero(torch.where(theta == closest_theta, 1, 0))
            loc = torch.argmax(target[locs[:, 0]][:, 0])

            name = [target[locs[:, 0]][loc][i].item() for i in range(4)]

            f_path = "steelData/" + steel_type + "_"
            for j in range(4):
                f_path += str(name[j]) + "_"

            img = Image.fromarray(full_pat.data[i] * 127.5)
            img = img.resize((64, 64))
            img.convert('L').save(f_path + ".jpeg")
            
for steel_type in steel_types:
    mp, mp_lp = get_mps(steel_type)
    generate_batch(steel_type, mp, mp_lp, 1000)