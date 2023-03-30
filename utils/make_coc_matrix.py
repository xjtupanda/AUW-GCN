import numpy as np
import pandas as pd
'''
    11 ROI regions in total
    AU12: 0, 1
    AU4: 2
    AU14: 3, 4
    AU6: 5, 6
    AU2: 7, 8
    AU1: 9, 10
'''

'''
    3 ROI regions: brows and mouth
    AU12, 14: 2
    AU1,2 : 0, 1
'''
'''
    12 points by whcold.
    
    ior_flows = get_rois(
        flow, landmarks,
        indices=[
            18, 19, 20,     # left eyebrow
            23, 24, 25,     # right eyebrow
            28, 30,         # nose
            48, 51, 54, 57  # mouse
        ],
        horizontal_bound=radius,
        vertical_bound=radius
    )
    
    AU1: 0, 1, 4, 5     outer brow raiser
    AU2: 2, 3           inner brow raiser
    AU12, 14: 8, 10
    AU25, 26: 9, 11
'''
def update_single(mat, AU, AU_dict):
    au_idx_list = AU_dict[AU]
    for idx in au_idx_list:
        mat[idx][idx] += 1
    
    return mat

def update_double(mat, AU1, AU2, AU_dict):
    au_idx_list1 = AU_dict[AU1]
    au_idx_list2 = AU_dict[AU2]
    
    for idx in au_idx_list1:
        for jdx in au_idx_list2:
            mat[idx][jdx] += 1
            mat[jdx][idx] += 1
    return mat

def update_mat(mat, AU_list, AU_dict):
    if len(AU_list) == 1:
       mat = update_single(mat, AU_list[0], AU_dict)
    else:
       for idx in range(len(AU_list)):
           for jdx in range(idx + 1, len(AU_list)):
               mat = update_double(mat, AU_list[idx], AU_list[jdx], AU_dict)
    return mat

#label_path = '/data/xjtupanda/experiments/Detection/ME_spotting/SoftNet-SpotME-main/CASME_sq/label_final.csv'
label_path = '/data/xjtupanda/experiments/Detection/ME_spotting/SoftNet-SpotME-main/SAMMLV/label_final_new.csv'
df = pd.read_csv(label_path, dtype={'au':'string'})
# # make expression-type specific
# df = df[df['type'] == 'macro-expression']
AU = df.au.values


# AU1: 0, 1, 4, 5     outer brow raiser
#     AU2: 2, 3           inner brow raiser
#     AU12, 14: 8, 10
#     AU25, 26: 9, 11
    
# AU_idx_dict = {
#     '12': [0, 1],
#     '4':   [2],
#     '14': [3, 4],
#     '6': [5, 6],
#     '2': [7, 8],
#     '1': [9, 10]
# }
# AU_idx_dict = {
#     '12': [2],
#     '14': [2],
#     '2': [0, 1],
#     '1': [0, 1]
# }

AU_idx_dict = {
    '1': [0, 1, 4, 5],
    '2': [2, 3],
    '12': [8, 10],
    '14': [8, 10],
    '25': [9, 11],
    '26': [9, 11]
}

# #print(AU_idx_dict)
AU_total = []

co_matrix = np.zeros((12, 12), dtype=np.float32)
for row in AU:
    if pd.isna(row):continue
    row_AU_list = row.split("+")
    # filter out unwanted AUs
    row_AU_list = [i for i in row_AU_list if i in AU_idx_dict.keys()]
    co_matrix = update_mat(co_matrix, row_AU_list, AU_idx_dict)

norm = np.linalg.norm(co_matrix, ord=1, axis=1, keepdims=True) + 1e-6
co_matrix = np.divide(co_matrix, norm)
print(co_matrix)
np.save('./co_matrix_SAMM_12ROI.npy', co_matrix)
