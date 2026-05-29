import pandas as pd
import os
import numpy as np
import glob


file_path = "/userhome/distorted_PCs/"


pc_dirs = glob.glob(r"/userhome/distorted_PCs/*/*.ply")
#print(pc_dirs)
assert len(pc_dirs) == 400

selected_names = np.random.choice(pc_dirs, 400, replace=False)

train_names = selected_names[:300]
test_names = selected_names[-100:]
assert len(train_names) == 300 and len(test_names) == 100


mosfile = "/userhome/WPC2.0_MOS.xlsx"

mos = pd.read_excel(mosfile, index_col=None)

print(mos)

train_lines = []
test_lines = []

l = len(mos)
for i in range(l):
    content_name, filename, _, _, moslabel = mos.iloc[i]
    
    dir_path = os.path.join(file_path, content_name, filename)

    if not os.path.exists(dir_path):
        print(filename)
        print(dir_path)
        raise ValueError

    #print(filename)
    if dir_path in train_names:
        train_lines.append((filename, dir_path, moslabel))
    elif dir_path in test_names:
        test_lines.append((filename, dir_path, moslabel))
    else:
        raise ValueError


train_lines = pd.DataFrame(data=train_lines)
test_lines = pd.DataFrame(data=test_lines)

train_lines.to_csv("wpc20_train.txt", header=None, index=None)
test_lines.to_csv("wpc20_test.txt", header=None, index=None)