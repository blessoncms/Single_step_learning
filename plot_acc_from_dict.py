import json
import numpy as np
with open('resapr_30.json') as f:
    d = json.load(f)
    start='cy:'
    end='Lo'
    
    a=[np.float((''.join(map(str,d[key])).split(start))[1].split(end)[0])  for key in d]
   
from matplotlib import pyplot as plt
plt.ylabel('Accuracy')

plt.xlabel('No. of iterations')
plt.plot(a)
plt.show()
