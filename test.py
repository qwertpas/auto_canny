import time
import math


last = time.time()
for i in range (1,100000):
    print("spam" + str(math.log(i)))

print(time.time()-last)