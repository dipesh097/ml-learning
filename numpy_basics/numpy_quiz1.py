# x = np.array([2,5,8,11,14])
#
# Write code to:
#
# Replace all values greater than 10 with 999
# Replace all even numbers with 0
import numpy as np

x=np.array([2,3,4,11,14])

x[x>10]=999
print(x)

x[x%2==0]=0
print(x)