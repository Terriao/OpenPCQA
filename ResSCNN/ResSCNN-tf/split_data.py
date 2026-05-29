import pandas as pd
import os
import numpy as np


lossname2dirname = {
    "limitlossyG-lossyA": "AVS_limitlossyG-lossyA",
    "losslessG-limitlossyA": "AVS_losslessG-limitlossyA",
    "losslessG-lossyA": "AVS_losslessG-lossyA",
    
}

def lossname2dirname(loss_name):
    ret_name = loss_name
    if loss_name in ["limitlossyG-lossyA", "losslessG-limitlossyA", "losslessG-lossyA"]:
        ret_name = "AVS_" + ret_name
    elif loss_name in ["lossless-geom-lossy-attrs", "lossless-geom-nearlossless-attrs", "lossy-geom-lossy-attrs"]:
        ret_name = "GPCC_" + ret_name
    elif loss_name == "C2AI-lossy-geom-lossy-attrs":
        ret_name = "VPCC_lossy-geom-lossy-attrs"
    elif loss_name == "Octree":
        ret_name = "octree"
    
    return ret_name


path = "/home/lhh/datasets/LS-PCQA/Distortion/AVS_limitlossyG-lossyA"

names = []

for file in os.listdir(path):
    name = file.strip().split("_")[0]
    if name not in names:
        names.append(name)

selected_names = np.random.choice(names, 104, replace=False)

train_names = selected_names[:100]
test_names = selected_names[-4:]


mosfile = "./config/subjectiveMOS.xlsx"

mos = pd.read_excel(mosfile, index_col=None)

print(mos)

train_lines = []
test_lines = []

l = len(mos)
for i in range(l):
    filename, moslabel = mos.iloc[i]

    part_fn = filename.strip().split("_")
    content_name = part_fn[0]
    dir_path = lossname2dirname(part_fn[1])

    if not os.path.exists(os.path.join("/home/lhh/datasets/LS-PCQA/Distortion", dir_path)):
        print(filename)
        print(dir_path)
        raise ValueError

    if content_name in train_names:
        train_lines.append((filename, dir_path, moslabel))
    elif content_name in test_names:
        test_lines.append((filename, dir_path, moslabel))
    else:
        raise ValueError


train_lines = pd.DataFrame(data=train_lines)
test_lines = pd.DataFrame(data=test_lines)

train_lines.to_csv("ls-pcqa_train.txt", header=None, index=None)
test_lines.to_csv("ls-pcqa_test.txt", header=None, index=None)