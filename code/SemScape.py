import json
from scipy.spatial import distance_matrix
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_array 

f = open('/path/to/your/VisualGenome_objects.json') # get from http://visualgenome.org

# returns JSON object as a dictionary
data = json.load(f)
len(data) # 108,077 images

all_objects = []
for dat in data:
    for obj in dat["objects"]:
        all_objects.append(obj["names"][0])
print(len(all_objects)) # 2,516,939 object tokens
print(len(set(all_objects))) # 82,827 object types

# to be used later on
word_dict = {num: word for num, word in enumerate(set(all_objects))}
idx_dict = {value:key for key, value in word_dict.items()}

#######################################################################
# distances as normalized ranks (x = (r-1)/(R-1)) where R == max rank #
#######################################################################

def sim_rank(img): # from objects, find (x, y) coordinates, then distance matrix, then similarity rank matrix (with diagonal set to 0)
    out = []
    idx = []
    for obj in img["objects"]:
        new_y = obj["y"]+(obj["h"]/2)
        new_x = obj["x"]+(obj["w"]/2)
        name = obj["names"][0]
        out.append([new_x, new_y])
        idx.append(name)
    df = pd.DataFrame(out, columns=["xcord", "ycord"], index=idx)
    sim = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index).rank(axis=0, ascending=False) # Euclidean distance matrix
    maxrank = max(sim.max())
    sim = (sim-1)/(maxrank-1)
    np.fill_diagonal(sim.values, 0) # diagonal to 0
    return sim


M = csr_array((len(set(all_objects)), len(set(all_objects))), dtype=np.float32).toarray() # create empty matrix, Obj_types X Obj_types

# filling the matrix with rank-normalized distances
for pic in tqdm(data):
    if pic["objects"]: # remove empty pictures
        sim_df = sim_rank(pic)
        for col_idx, colname in enumerate(sim_df.columns):
            for idx_idx, indexname in enumerate(sim_df.index):
                dist = sim_df.iloc[idx_idx, col_idx] # distance between (idx, col)
                M[idx_dict[indexname], idx_dict[colname]] += dist

M = sp.csr_matrix(M) # convert to sparse to speed up calculations

U_vg, s_vg, VT_vg = svds(M, k = 25)
print(U_vg.shape, s_vg.shape, VT_vg.shape)

d_vizgen_rank = {}
for n, word in word_dict.items():
    vec = U_vg[n, :]
    d_vizgen_rank[word] = vec

with open("vizgen_rank_25d", 'wb') as f:
    pickle.dump(d_vizgen_rank, f)
