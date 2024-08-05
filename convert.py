import os
import re

PATH = "/home/user/Downloads"

files = os.listdir(PATH)
ply_files = []

for f in files:
    if re.search(".stl$", f) is None:
        continue
    ply_files.append(f)

for f in ply_files:
    n, t, j = os.path.basename(f).split(" ")
    if j == "LowerJawScan.stl":
        j = "lower"
    else:
        j = "upper"

    os.rename(os.path.join(PATH, f), os.path.join(PATH, f"{n}-{t}_{j}.stl"))
