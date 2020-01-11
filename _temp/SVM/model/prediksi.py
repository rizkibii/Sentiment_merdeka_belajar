import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = "/".join(dir_path.split('/')[:-1])
# print(dir_path)
# dir_path = os.chdir("../"+dir_path)
sys.path.append(dir_path)
# print(sys.path)

import praproses.praproses as pps 

print(pps.praproses2("memakan jahatt", n=1))