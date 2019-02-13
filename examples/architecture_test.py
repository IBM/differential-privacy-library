import sys
from os.path import abspath

sys.path.append(abspath('..'))

# print(abspath('.'))

import diffprivlib.mechanisms as mechs

mech = mechs.Laplace()

print("DPMachine: " + str(type(mech)))
