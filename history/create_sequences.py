import numpy as np
import pandas as pd
import sys

patid = int(sys.argv[1])
block_num = int(sys.argv[2])

patterns = {1:[4,2,3,1], 2:[2,4,1,3], 
            3:[1,3,4,2], 4:[3,1,2,4]}


# Repeat the pattern 10 times
pat = patterns[patid]
ori_pattern = np.tile(pat, 10*block_num)
ori_random = np.tile(pat, 10*block_num)

# Shuffle the sequence randomly
np.random.shuffle(ori_random)

# Initialize the result array
ori_seq = np.zeros(len(ori_pattern) + len(ori_random), dtype=int)

# Interleave the arrays
ori_seq[::2] = ori_pattern
ori_seq[1::2] = ori_random
# Create a DataFrame to store ori_random and ori_pattern
df = pd.DataFrame({
    'orientation':ori_seq*90,
    'orientation_id':ori_seq*90
    })
df.to_csv('sequences/learning_sequence.csv', index = False)