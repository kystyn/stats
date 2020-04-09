import part3
import utils
import numpy as np

d = np.sort(utils.get_distribution('Normal', 100))
s = 0
for i in d:
    s += i
print(s)

#part3.lab3()
