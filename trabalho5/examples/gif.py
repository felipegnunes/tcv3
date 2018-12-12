import imageio
import sys
import os

import numpy as np

l = [os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1], f))]
l.sort(key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))
images = []

for filename in l:
	images.append(imageio.imread(filename))

imageio.mimsave('gan.gif', images, duration = 0.25)
