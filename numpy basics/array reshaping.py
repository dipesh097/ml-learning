import numpy as np
arr=np.array([
              [1,3,5],
                [4,6,8],
                 [2,6,7]
            ])

 # print(arr.flatten())

# # --------------------------- masking------------------------------------------
# print([arr>3])
# print(arr[arr<5])

arr[arr>3]=99
print(arr)