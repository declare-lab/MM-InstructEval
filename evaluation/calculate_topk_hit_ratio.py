import numpy as np
import matplotlib.pyplot as plt
import heapq
fig, ax = plt.subplots()      # 子图
from collections import Counter
import numpy as np

# 封装一下这个函数，用来后面生成数据
def list_generator(mean, dis, number):
    return np.random.normal(mean, dis * dis, number)  # normal分布，输入的参数是均值、方差以及生成的数量


# 我们生成四组数据用来做实验，数据量分别为70-100
# 分别代表男生、女生在20岁和30岁的花费分布

def get_max_nums(topk, result_list):
    max_num_index_list_new = []
    # max_num_index_list = list(map(result_list.index, heapq.nlargest(max_nums, result_list)))
    max_num_index_list = [i for _, i in sorted(zip(result_list, range(len(result_list))), reverse=True)[:topk]]
    for i in range(len(max_num_index_list)):
        max_num_index_list_new.append(max_num_index_list[i]+1)
    return max_num_index_list_new

###ChatGPT
MVSA_Single_gpt = [48.301, 49.272, 56.553, 46.602, 54.612, 43.358, 54.094, 36.842, 7.519, 44.11]
Twitter_2015_gpt = [61.716, 64.609, 56.027, 65.477, 63.067, 61.486, 63.067, 60.714, 60.714, 63.873]
Hate_gpt = [57.7, 58.043, 55.459, 35.808, 59.389, 52.759, 59.389, 60.838, 54.86, 58.78]
MSD_gpt = [66.253, 68.948, 52.067, 63.925, 34.739, 62.965, 69.019, 62.463, 64.676, 60.919]
MNRE_gpt = [37.031, 38.281, 16.719, 34.063, 21.563, 20.469, 27.259, 32.344, 32.031, 28.438]
ScienceQA_gpt = [59.871, 62.351, 67.229, 61.26, 68.121, 57.242, 69.41, 60.218, 58.602, 56.746]


##llama1-7b
MVSA_Single_llama1_7b = [67.23, 55.34, 8.981, 0.243, 8.981, 20.874, 45.146, 39.563, 11.165, 0]
Twitter_2015_llama1_7b = [30.569, 29.508, 58.534, 0.675, 58.534, 22.951, 25.844, 4.243, 11.861, 0.193]
Hate_llama1_7b = [40, 48.8, 48.2, 50.4, 49.2, 33, 48.8, 37.6, 0.2, 24.2]
MSD_llama1_7b = [43.67, 50.021, 36.364, 58.987, 39.809, 52.719, 58.032, 25.529, 0.457,0.332]
MNRE_llama1_7b = [1.094, 0.469, 2.656, 0, 0.156, 0.156, 0.313, 0.156, 0.469, 1.094]
ScienceQA_llama1_7b = [9.618, 33.069, 0.149, 21.666, 36.192, 17.997, 32.375, 12.692, 19.683, 23.401]


##llama1-13b
MVSA_Single_llama1_13b = [64.806, 66.99, 9.223, 4.612, 60.437, 59.951, 46.602, 62.864, 16.505, 3.883]
Twitter_2015_llama1_13b = [28.544, 29.508, 52.073, 9.161, 30.569, 31.919, 37.608, 24.783, 15.526, 21.697]
Hate_llama1_13b = [33, 33.6, 44.8, 47.2, 49.2, 12.6, 47.8, 21.8, 11.6, 46.4]
MSD_llama1_13b = [55.168, 36.903, 27.356, 57.534, 32.503, 11.166, 13.2, 29.722, 20.299, 27.937]
MNRE_llama1_13b = [10.156, 0.469, 0, 0, 19.219, 0.313, 14.063, 0.156, 0.313, 0.781]
ScienceQA_llama1_13b = [9.569, 26.376, 22.558, 4.313, 36.837, 43.332, 34.408, 3.52, 24.492, 7.784]



##llama2-7b
MVSA_Single_llama2_7b = [66.99, 63.835, 8.981, 39.563, 42.718, 48.301, 39.32, 31.311, 51.699, 22.816]
Twitter_2015_llama2_7b = [27.483, 34.33, 58.534, 5.4, 33.365, 30.376, 26.422, 0, 22.179, 0.1]
Hate_llama2_7b = [52, 47.6, 49.2, 36.6, 49.2, 1, 34.2, 39.4, 2.6, 6.8]
MSD_llama2_7b = [56.33, 36.571, 39.809, 55.376, 39.851, 38.439, 37.817, 32.379, 29.763, 7.846]
MNRE_llama2_7b = [2.188, 1.094, 2.813, 0, 2.813, 0, 3.594, 0.156, 0.156, 0]
ScienceQA_llama2_7b = [19.881, 30.243, 30.144, 22.062, 39.465, 43.084, 41.448, 11.651, 26.772, 3.57]


##llama2-13b
MVSA_Single_llama2_13b = [66.019, 63.835, 1.699, 0, 34.223, 52.427, 57.282, 58.495, 18.447, 19.903]
Twitter_2015_llama2_13b = [31.823, 31.919, 29.701, 2.989, 60.366, 31.823, 40.116, 0.096, 21.601, 0]
Hate_llama2_13b = [51, 55, 1.2, 43.8, 43.8, 9.8, 53.8, 44, 3.2, 4.8]
MSD_llama2_13b = [60.232, 54.213, 1.038, 58.904, 41.926, 9.714, 52.387, 36.405, 18.306, 7.14]
MNRE_llama2_13b = [1.25, 0.313, 0, 0, 20, 0, 10.625, 0.156, 0, 0]
ScienceQA_llama2_13b = [4.165, 20.079, 30.59, 1.438, 40.208, 45.761, 55.776, 5.354, 30.987, 3.024]


##flant5
MVSA_Single_flant5 = [63.835, 63.592, 49.757, 64.806, 52.184, 62.621, 64.078, 63.835, 29.126, 64.078]
Twitter_2015_flant5  = [47.734, 47.541, 71.745, 47.637, 72.131, 47.927, 52.17, 49.952, 45.13, 47.927]
Hate_flant5  = [55, 56.2, 55.6, 54.8, 55.6, 56.6, 57.4, 54.6, 57.2, 56.4]
MSD_flant5  = [66.625, 70.32, 68.618, 69.116, 69.033, 70.901, 69.738, 65.546, 71.399, 70.237]
MNRE_flant5  = [25.625, 29.688, 30.938, 28.125, 31.25, 29.063, 30.469, 24.688, 31.406, 29.219]
ScienceQA_flant5 = [65.592, 66.634, 66.287, 66.138, 67.229, 65.791, 65.84, 66.584, 67.427, 66.237]


##openflamingo
ScienceQA_openflamingo = [7.933, 27.119, 23.748, 3.718, 39.266, 21.429, 33.118, 6.148, 20.526, 3.223]
MVSA_Single_openflamingo = [54.126, 37.379, 4.369, 1.942, 8.981, 11.65, 55.583, 52.427, 2.427, 0.485]
Twitter_2015_openflamingo  = [32.98, 20.058, 8.1, 1.929, 57.281, 17.3578, 38.091, 0.386, 8.197, 1.35]
Hate_openflamingo  = [34, 49.4, 20.8, 4.8, 24.6, 25, 31.6, 9.2, 47.6, 1.8]
MSD_openflamingo  = [52.677, 49.398, 7.098, 32.711, 1.37, 30.095, 34.122, 11.789, 28.601, 1.079]
MNRE_openflamingo  = [1.094, 1.719, 0.313, 0.938, 0, 3.125, 0, 0, 0.938, 0.625]



###Formage
MVSA_Single_fromage = [20.146, 28.641, 3.883, 1.456, 8.01, 29.854, 21.117, 1.942, 10.942, 0.243]
Twitter_2015_fromage  = [1.061, 18.804, 12.825, 0.096, 3.086, 19.961, 5.014, 0.193, 3.279, 0.096]
Hate_fromage  = [23, 37.6, 1, 18.2, 21.6, 31.2, 32.8, 8, 4.6, 0.6]
MSD_fromage  = [22.665, 39.311, 1.494, 11.291, 34.247, 39.975, 40.681, 14.902, 7.721, 2.242]
MNRE_fromage  = [0.156, 0.156, 0, 0, 0, 0.156, 0.156, 0, 0, 0]
ScienceQA_fromage = [6.743, 20.129, 14.08, 5.503, 30.144, 11.552, 34.507, 3.123, 10.61, 1.686]



###LLaVA-7B
MVSA_Single_llava7b = [37.864, 45.631, 55.097, 39.32, 53.641, 47.087, 56.553, 45.388, 45.874, 38.35]
Twitter_2015_llava7b  = [23.433, 20.926, 27.001, 19.865, 26.519, 23.819, 28.255, 21.697, 22.469, 22.372]
Hate_llava7b  = [9.6, 14.4, 20, 7.8, 12, 8.2, 20.2, 22.8, 16.6, 8]
MSD_llava7b  = [12.121, 8.302, 10.751, 7.97, 12.08, 6.932, 9.672, 13.45, 10.419, 11.955]
MNRE_llava7b  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ScienceQA_llava7b = [35.25, 38.721, 33.416, 34.209, 41.101, 38.076, 40.307, 36.192, 36.837, 36.093]


###LLaVA-13B
MVSA_Single_llava13b = [45.631, 56.311, 58.01, 54.126, 56.553, 55.097, 53.398, 36.65, 53.398, 42.961]
Twitter_2015_llava13b  = [27.387, 27.29, 28.351, 26.133, 27.965, 26.326, 27.001, 26.905, 27.29, 27.194]
Hate_llava13b  = [18, 28, 12.2, 28.4, 16, 24.4, 16, 12.2, 18, 15.4]
MSD_llava13b  = [21.046, 17.435, 15.235, 28.726, 16.853, 21.171, 24.616, 15.899, 12.536, 15.442]
MNRE_llava13b  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ScienceQA_llava13b = [37.481, 38.225, 37.134, 44.175, 47.496, 36.986, 47.744, 37.779, 32.474, 36.49]


## MiniGPT4
MVSA_Single_miniGPT4 = [30.583, 62.379, 68.447, 31.068, 71.12, 58.98, 47.573, 44.417, 41.262, 47.573]
Twitter_2015_miniGPT4  = [13.018, 35.583, 35.68, 13.115, 37.99, 37.99, 29.412, 8.293, 9.354, 29.605]
Hate_miniGPT4  = [41.4, 45.4, 1.6, 45.8, 20.8, 44.2, 35.2, 48.4, 47.8, 45.8]
MSD_miniGPT4  = [17.31, 37.65, 0.872, 37.526, 10.502, 31.465, 38.398, 31.963, 1.725, 38.149]
MNRE_miniGPT4  = [0.469, 1.719, 1.25, 2.031, 1.094, 2.813, 0.781, 0.938, 1.094, 2.188]
ScienceQA_miniGPT4  = [36.292, 46.108, 54.735, 44.175, 58.701, 43.629, 56.123, 41.894, 46.901, 43.927]


# mPLUG-Owl
MVSA_Single_mPLUG_OWL = [53.883, 52.427, 33.98, 34.46, 31.31, 32.524, 32.524, 29.368, 46.116, 42.718]
Twitter_2015_mPLUG_OWL = [30.086, 36.354, 21.504, 8.678, 28.447, 16.682, 30.183, 13.982, 20.347, 17.261]
Hate_mPLUG_OWL = [49, 16.4, 35.6, 22.2, 44.8, 45.4, 49.2, 24.2, 49.2, 47.2]
MSD_mPLUG_OWL = [37.069, 36.529, 28.933, 24.491, 36.488, 36.031, 39.767, 19.302, 39.601, 32.254]
MNRE_mPLUG_OWL = [1.875, 3.437, 6.093, 2.5, 8.75, 3.75, 3.281, 3.906, 3.437, 2.656]
ScienceQA_mPLUG_OWL = [14.626, 31.532, 27.268, 15.121, 32.871, 37.481, 37.928, 23.352, 30.987, 9.668]


# AdapterV2
MVSA_Single_Adapterv2 = [46.117, 54.612, 59.951, 53.155, 59.951, 30.825, 52.427, 42.476, 53.641, 47.816]
Twitter_2015_Adapterv2  = [24.976, 26.037, 30.473, 27.387, 30.569, 8.582, 21.89, 23.722, 28.447, 25.651]
Hate_Adapterv2  = [46.6, 49, 44.8, 49, 46, 2.8, 0.2, 45.4, 13.8, 46.2]
MSD_Adapterv2  = [13.616, 39.601, 1.785, 39.768, 16.438, 8.593, 18.099, 35.201, 35.077, 31.963]
MNRE_Adapterv2  = [0, 0, 0, 0, 0, 0, 0.156, 0, 0, 0]
ScienceQA_Adapterv2 = [49.876, 49.479, 46.802, 48.587, 49.777, 43.48, 54.437, 49.033, 49.579, 50.719]


# VPGTrans
MVSA_Single_VPGTrans = [31.311, 24.272, 58.495, 51.942, 58.495, 39.078, 59.223, 23.544, 44.903, 24.757]
Twitter_2015_VPGTrans  = [18.901, 12.633, 26.23, 16.393, 28.351, 23.722, 24.012, 9.547, 17.454, 5.014]
Hate_VPGTrans  = [1.8, 4.4, 44, 11.4, 45.2, 8.8, 38.2, 8.2, 15.2, 20.4]
MSD_VPGTrans  = [14.238, 9.423, 8.925, 12.412, 32.462, 15.401, 30.137, 13.242, 9.381, 23.827]
MNRE_VPGTrans  = [0, 0.156, 0.313, 0.156, 0.156, 0, 0.156, 0, 0.156, 0]
ScienceQA_VPGTrans = [17.253, 34.755, 47, 36.886, 39.018, 18.493, 35.3, 15.766, 35.201, 23.599]


# MultiGPT
MVSA_Single_MultiGPT = [50, 48.544, 1.942, 0.243, 8.981, 26.942, 52.913, 0, 37.379, 7.524]
Twitter_2015_MultiGPT  = [32.883, 19.383, 20.347, 9.45, 58.534, 26.905, 30.665, 0, 18.901, 0.482]
Hate_MultiGPT  = [40.8, 49.8, 47.2, 5, 48, 33, 31.8, 0, 48.2, 2.4]
MSD_MultiGPT  = [54.711, 58.78, 11.166, 59.817, 4.068, 44.666, 35.741, 0, 28.477, 13.159]
MNRE_MultiGPT  = [2.813, 2.344, 2.813, 0, 0, 0.156, 0, 0, 0.469, 0.469]
ScienceQA_MultiGPT = [6.247, 23.302, 23.302, 9.569, 36.292, 22.459, 25.781, 5.354, 19.931, 3.917]


# LaVIN-7B
MVSA_Single_LaVIN_7B = [19.417, 39.32, 20.388, 31.068, 14.563, 33.738, 30.583, 14.806, 25, 34.951]
Twitter_2015_LaVIN_7B  = [37.223, 27.483, 23.337, 25.94, 11.765, 13.211, 15.526, 8.004, 11.379, 15.622]
Hate_LaVIN_7B  = [44.8, 43.6, 38.2, 47.2, 45.6, 35.2, 50.4, 26.8, 21.8, 44.6]
MSD_LaVIN_7B  = [56.953, 54.711, 45.413, 56.164, 59.527, 25.073, 60.482, 33.001, 41.926, 59.112]
MNRE_LaVIN_7B  = [0.156, 2.656, 6.563, 0.156, 12.344, 0.156, 1.094, 1.25, 0.469, 0.313]
ScienceQA_LaVIN_7B = [4.511, 53, 75.112, 37.779, 74.467, 15.007, 51.71, 17.898, 57.362, 21.705]


# LaVIN-13B
MVSA_Single_LaVIN_13B = [44.417, 48.544, 10.437, 53.641, 11.893, 51.214, 13.35, 28.641, 14.806, 39.078]
Twitter_2015_LaVIN_13B  = [17.647, 34.426, 17.551, 24.976, 8.679, 35.391, 33.462, 7.136, 16.393, 17.454]
Hate_LaVIN_13B  = [49.6, 41.2, 29.2, 28.4, 24, 35.2, 48.8, 23.2, 29.4, 44.2]
MSD_LaVIN_13B  = [55.832, 31.465, 30.345, 38.315, 30.594, 36.737, 57.576, 20.008, 15.567, 35.035]
MNRE_LaVIN_13B  = [11.563, 3.125, 3.125, 1.406, 5.938, 5, 9.063, 3.594, 2.188, 2.031]
ScienceQA_LaVIN_13B = [30.491, 53.297, 60.387, 17.97, 77.541, 31.532, 56.668, 12.097, 48.637, 33.845]


# Lynx
MVSA_Single_Lynx = [41.505, 61.408, 47.087, 29.126, 53.155, 50.243, 64.32, 5.34, 59.709, 56.796]
Twitter_2015_Lynx  = [29.219, 34.137, 35.005, 33.655, 30.473, 45.998, 31.726, 2.604, 40.212, 42.527]
Hate_Lynx  = [38.6, 51.2, 7.2, 50, 5.4, 32, 51.6, 12, 42, 46.2]
MSD_Lynx  = [40.349, 34.952, 15.899, 36.779, 11.706, 31.507, 43.96, 11.706, 22.499, 32.586]
MNRE_Lynx  = [0, 2.188, 4.844, 2.969, 1.25, 8.438, 1.094, 3.281, 9.219, 1.875]
ScienceQA_Lynx = [16.559, 18.989, 17.551, 0.744, 29.698, 18.195, 38.275, 7.238, 34.136, 27.714]


# BLIP-2
MVSA_Single_BLIP_2 = [64.32, 64.563, 51.7, 66.262, 51.213, 64.32, 66.019, 64.32, 59.951, 65.777]
Twitter_2015_BLIP_2  = [46.48, 47.637, 69.335, 46.287, 70.781, 47.541, 48.505, 48.505, 47.059, 48.312]
Hate_BLIP_2  = [55.8, 56.8, 56.6, 56.6, 56.6, 56.6, 58, 56.6, 57, 55.2]
MSD_BLIP_2  = [67.538, 72.022, 68.95, 71.274, 70.237, 71.357, 70.818, 67.538, 70.776, 71.15]
MNRE_BLIP_2  = [34.688, 33.125, 29.844, 30.625, 30.781, 30.156, 30, 28.75, 28.438, 32.656]
ScienceQA_BLIP_2  = [74.17, 73.82, 71.195, 73.872, 72.434, 62.766, 72.286, 56.073, 73.475, 73.426]


# InstructBLIP
MVSA_Single_InstructBLIP = [63.835, 68.932, 61.165, 66.99, 58.981, 71.602, 68.447, 62.136, 69.417, 70.631]
Twitter_2015_InstructBLIP  = [49.952, 43.684, 62.199, 46.962, 63.067, 46.191, 43.394, 55.545, 45.709, 45.902]
Hate_InstructBLIP  = [57.2, 58, 57, 56.2, 56.8, 56.4, 56.2, 57.2, 58.2, 57.2]
MSD_InstructBLIP  = [69.365, 71.689, 72.146, 72.146, 72.976, 70.32, 73.101, 70.652, 71.856, 69.531]
MNRE_InstructBLIP  = [30.469, 36.719, 29.375, 35.938, 32.5, 35.156, 31.406, 23.125, 34.375, 35.469]
ScienceQA_InstructBLIP = [71.691, 73.327, 73.079, 72.484, 72.93, 71.74, 70.749, 71.889, 72.335, 72.335]


all_chatGPT_results = {
    'ScienceQA': ScienceQA_gpt,
    'MVSA_Single': MVSA_Single_gpt,
    'Twitter_2015': Twitter_2015_gpt,
    'Hate': Hate_gpt,
    'MSD': MSD_gpt,
    'MNRE': MNRE_gpt,
    
}

all_llama1_7b_results = {
    'ScienceQA': ScienceQA_llama1_7b,
    'MVSA_Single': MVSA_Single_llama1_7b,
    'Twitter_2015': Twitter_2015_llama1_7b,
    'Hate': Hate_llama1_7b,
    'MSD': MSD_llama1_7b,
    'MNRE': MNRE_llama1_7b,
}

all_llama1_13b_results = {
    'ScienceQA': ScienceQA_llama1_13b,
    'MVSA_Single': MVSA_Single_llama1_13b,
    'Twitter_2015': Twitter_2015_llama1_13b,
    'Hate': Hate_llama1_13b,
    'MSD': MSD_llama1_13b,
    'MNRE': MNRE_llama1_13b,
    
}

all_llama2_7b_results = {
    'ScienceQA': ScienceQA_llama2_7b,
    'MVSA_Single': MVSA_Single_llama2_7b,
    'Twitter_2015': Twitter_2015_llama2_7b,
    'Hate': Hate_llama2_7b,
    'MSD': MSD_llama2_7b,
    'MNRE': MNRE_llama2_7b,
}

all_llama2_13b_results = {
    'ScienceQA': ScienceQA_llama2_13b,
    'MVSA_Single': MVSA_Single_llama2_13b,
    'Twitter_2015': Twitter_2015_llama2_13b,
    'Hate': Hate_llama2_13b,
    'MSD': MSD_llama2_13b,
    'MNRE': MNRE_llama2_13b,
}

all_flant5_results = {
    'ScienceQA': ScienceQA_flant5,
    'MVSA_Single': MVSA_Single_flant5,
    'Twitter_2015': Twitter_2015_flant5,
    'Hate': Hate_flant5,
    'MSD': MSD_flant5,
    'MNRE': MNRE_flant5,
}




all_openflamingo_results = {
     'ScienceQA': ScienceQA_openflamingo,
    'MVSA_Single': MVSA_Single_openflamingo,
    'Twitter_2015': Twitter_2015_openflamingo,
    'Hate': Hate_openflamingo,
    'MSD': MSD_openflamingo,
    'MNRE': MNRE_openflamingo,
   
}



all_fromage_results = {
    'ScienceQA': ScienceQA_fromage,
    'MVSA_Single': MVSA_Single_fromage,
    'Twitter_2015': Twitter_2015_fromage,
    'Hate': Hate_fromage,
    'MSD': MSD_fromage,
    'MNRE': MNRE_fromage,
    
}



all_llava7b_results = {
    'ScienceQA': ScienceQA_llava7b,
    'MVSA_Single': MVSA_Single_llava7b,
    'Twitter_2015': Twitter_2015_llava7b,
    'Hate': Hate_llava7b,
    'MSD': MSD_llava7b,
    'MNRE': MNRE_llava7b,
}


all_llava13b_results = {
    'ScienceQA': ScienceQA_llava13b,
    'MVSA_Single': MVSA_Single_llava13b,
    'Twitter_2015': Twitter_2015_llava13b,
    'Hate': Hate_llava13b,
    'MSD': MSD_llava13b,
    'MNRE': MNRE_llava13b,
    
}


all_miniGPT4_results = {
    'ScienceQA': ScienceQA_miniGPT4,
    'MVSA_Single': MVSA_Single_miniGPT4,
    'Twitter_2015': Twitter_2015_miniGPT4,
    'Hate': Hate_miniGPT4,
    'MSD': MSD_miniGPT4,
    'MNRE': MNRE_miniGPT4,
    
}



all_mPLUG_OWL_results = {
    'ScienceQA': ScienceQA_mPLUG_OWL,
    'MVSA_Single': MVSA_Single_mPLUG_OWL,
    'Twitter_2015': Twitter_2015_mPLUG_OWL,
    'Hate': Hate_mPLUG_OWL,
    'MSD': MSD_mPLUG_OWL,
    'MNRE': MNRE_mPLUG_OWL,
    
}



all_Adapterv2_results = {
    'ScienceQA': ScienceQA_Adapterv2,
    'MVSA_Single': MVSA_Single_Adapterv2,
    'Twitter_2015': Twitter_2015_Adapterv2,
    'Hate': Hate_Adapterv2,
    'MSD': MSD_Adapterv2,
    'MNRE': MNRE_Adapterv2,
    
}


all_VPGTrans_results = {
    'ScienceQA': ScienceQA_VPGTrans,
    'MVSA_Single': MVSA_Single_VPGTrans,
    'Twitter_2015': Twitter_2015_VPGTrans,
    'Hate': Hate_VPGTrans,
    'MSD': MSD_VPGTrans,
    'MNRE': MNRE_VPGTrans,
    
}



all_MultiGPT_results = {
    'ScienceQA': ScienceQA_MultiGPT,
    'MVSA_Single': MVSA_Single_MultiGPT,
    'Twitter_2015': Twitter_2015_MultiGPT,
    'Hate': Hate_MultiGPT,
    'MSD': MSD_MultiGPT,
    'MNRE': MNRE_MultiGPT,
    
}



all_LaVIN_7B_results = {
    'ScienceQA': ScienceQA_LaVIN_7B,
    'MVSA_Single': MVSA_Single_LaVIN_7B,
    'Twitter_2015': Twitter_2015_LaVIN_7B,
    'Hate': Hate_LaVIN_7B,
    'MSD': MSD_LaVIN_7B,
    'MNRE': MNRE_LaVIN_7B,
    
}



all_LaVIN_13B_results = {
    'ScienceQA': ScienceQA_LaVIN_13B,
    'MVSA_Single': MVSA_Single_LaVIN_13B,
    'Twitter_2015': Twitter_2015_LaVIN_13B,
    'Hate': Hate_LaVIN_13B,
    'MSD': MSD_LaVIN_13B,
    'MNRE': MNRE_LaVIN_13B,
    
}


all_Lynx_results = {
    'ScienceQA': ScienceQA_Lynx,
    'MVSA_Single': MVSA_Single_Lynx,
    'Twitter_2015': Twitter_2015_Lynx,
    'Hate': Hate_Lynx,
    'MSD': MSD_Lynx,
    'MNRE': MNRE_Lynx,
    
}



all_BLIP_2_results = {
    'ScienceQA': ScienceQA_BLIP_2,
    'MVSA_Single': MVSA_Single_BLIP_2,
    'Twitter_2015': Twitter_2015_BLIP_2,
    'Hate': Hate_BLIP_2,
    'MSD': MSD_BLIP_2,
    'MNRE': MNRE_BLIP_2,
    
}

all_InstructBLIP_results = {
    'ScienceQA': ScienceQA_InstructBLIP,
    'MVSA_Single': MVSA_Single_InstructBLIP,
    'Twitter_2015': Twitter_2015_InstructBLIP,
    'Hate': Hate_InstructBLIP,
    'MSD': MSD_InstructBLIP,
    'MNRE': MNRE_InstructBLIP,
    
}


model_results_dict = {
    'chatGPT': all_chatGPT_results,
    'LLaMA1_7B': all_llama1_7b_results,
    'LLaMA1_13B': all_llama1_13b_results,
    'LLaMA2_7B': all_llama2_7b_results,
    'LLaMA2_13B': all_llama2_13b_results,
    'flanT5': all_flant5_results,
    'openFlamingo': all_openflamingo_results,
    'Fromage': all_fromage_results,
    'LLaVA_7B': all_llava7b_results,
    'LLaVA_13B': all_llava13b_results,
    'MiniGPT4': all_miniGPT4_results,
    'mPLUG_OWL': all_mPLUG_OWL_results,
    'Adapterv2': all_Adapterv2_results,
    'VPGTrans': all_VPGTrans_results,
    "MultiGPT": all_MultiGPT_results,
    "LaVIN_7B": all_LaVIN_7B_results,
    "LaVIN_13B": all_LaVIN_13B_results,
    'Lynx': all_Lynx_results,
    'BLIP_2': all_BLIP_2_results,
    'InstructBLIP': all_InstructBLIP_results
    
}


def get_all_index_list(topk, dataset_list, all_result_dict, model):
    all_topk_dict = {}
    result_dict = all_result_dict[model]
    for dataset in dataset_list:
        result = result_dict[dataset]
        max_num_index_list = get_max_nums(topk, result)
        all_topk_dict[dataset] = max_num_index_list
    return all_topk_dict

def calculate_hit_ratio(all_topk_dict, model):
    all_topk_list = []
    hit_ratio_dict = {1:0, 2:0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10:0}
    for topk_index in all_topk_dict.values():
        for topk in topk_index:
            all_topk_list.append(topk)
    # print(all_topk_list)
    
    all_length = len(all_topk_list)
    print(all_length)
    index_count_dict = Counter(all_topk_list)
    for key, value in index_count_dict.items():
        hit_ratio_dict[key] = value/all_length*100
    # print(f'For the {model}, index_count_dict is {index_count_dict}')
    # print(f'For the {model}, hit_ratio_dict is {hit_ratio_dict}')
    hit_ratio_list = []
    for hit_ratio in hit_ratio_dict.values():
        hit_ratio_list.append(round(hit_ratio, 2))
    print(f'For the {model}, hit_ratio_list is {hit_ratio_list}')
    model_std  = np.var(hit_ratio_list)
    print(f'For the {model}, std is {model_std}')
    max = np.max(hit_ratio_list)
    print(f'For the {model}, the max value is {max}')
    return hit_ratio_list, model_std
    
    

if __name__ == "__main__":
    topk = 3
    dataset_list = [ 'ScienceQA', 'MVSA_Single',
    'Twitter_2015', 
    'Hate', 'MSD', 'MNRE']
    all_result_dict = model_results_dict
    model_list = ['chatGPT', 'LLaMA1_7B', 'LLaMA1_13B', 'LLaMA2_7B', 'LLaMA2_13B', 'flanT5', 
                  'openFlamingo', 'Fromage', 'LLaVA_7B', 'LLaVA_13B',  'MiniGPT4', 'mPLUG_OWL', 'Adapterv2', 'VPGTrans', 'MultiGPT',
                  'LaVIN_7B', 'LaVIN_13B', 'Lynx',
                  'BLIP_2', 'InstructBLIP']
    LLM_model_list = ['chatGPT', 'LLaMA1_7B', 'LLaMA1_13B', 'LLaMA2_7B', 'LLaMA2_13B', 'flanT5']
    MLLM_model_list = [
                  'openFlamingo', 'Fromage', 'LLaVA_7B', 'LLaVA_13B',  'MiniGPT4', 'mPLUG_OWL', 'Adapterv2', 'VPGTrans', 'MultiGPT',
                  'LaVIN_7B', 'LaVIN_13B', 'Lynx',
                  'BLIP_2', 'InstructBLIP']
    
    
    all_hit_ratio_list = []
    total_hit_ratio_list = []
    all_model_std = []
    for model in model_list:
        all_topk_dict = get_all_index_list(topk, dataset_list, all_result_dict, model)
        print(all_topk_dict)
        hit_ratio_list, std = calculate_hit_ratio(all_topk_dict, model)
        
        all_hit_ratio_list.append(hit_ratio_list)
        all_model_std.append(round(std, 2))
    all_hit_ratio_list = np.array(all_hit_ratio_list)
    print(all_hit_ratio_list)
    print(all_model_std)
    
    _, instruction_len = np.shape(all_hit_ratio_list)
    for i in range(instruction_len):
        total_hit_ratio_list.append(round(sum(all_hit_ratio_list[:, i]), 2))
    print(total_hit_ratio_list) 


   