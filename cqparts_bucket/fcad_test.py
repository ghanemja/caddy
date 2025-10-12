import sys
sys.path.append('/home/ec2-user/Documents/cad-optimizer/squashfs-root/usr/lib')

import importlib.util
import os

# Load FreeCAD module manually
spec = importlib.util.spec_from_file_location("FreeCAD", "/home/ec2-user/Documents/cad-optimizer/squashfs-root/usr/lib/FreeCAD.so")
FreeCAD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FreeCAD)

print(FreeCAD)

