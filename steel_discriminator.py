import re
import kikuchipy as kp
from fastai.vision.all import *

num = 5

steel_types = ["a", "f"] # do later, ["c", "s"]
n_steels = len(steel_types)

def get_mps(steel_type):
    mp = kp.load("steelH5s/" + steel_type + ".h5")
    return mp, mp.as_lambert()

def breakdown(f_path):
    pat = r'[^\\_]+(?=_)+.+[^_.jpeg]'  
    pat = re.compile(pat)  
    extracted = pat.search(str(f_path)).group()
    #print(extracted)
    single_target = [float(lbl) for lbl in extracted.split("_")[1:]]
    
    one_hot = [0] * n_steels
    one_hot[steel_types.index(extracted[0])] = 1
    return one_hot #+ single_target

breakdown("steelData\\f_0.9046037197113037_-0.018439430743455887_0.31547197699546814_-0.2860586643218994")
#print("start main")

if __name__ == '__main__':
    dblock = DataBlock(blocks=(ImageBlock, RegressionBlock), get_items = get_image_files, get_y = breakdown)
    dls = dblock.dataloaders("steelData/", batch_size = 32, shuffle = True, pin_memory = True, persistent_workers = True, device = torch.device("cuda"))
    arc = resnet18

    learn = vision_learner(dls, arc, n_out = n_steels, first_bn = True, 
                        concat_pool = True, pool = True)
    #learn.remove_cb(ProgressCallback)

    learn.loss = CrossEntropyLossFlat
    #learn.lr_find()

    learn.lr = 0.001
    learn.fit(10000)