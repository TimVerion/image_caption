import pandas as pd

# annotations = pd.read_table('data/results_20130124.token', sep='\t', header=None,
#                             names=['image', 'caption'])
# print(annotations.head())
# for i  in range(10):
#     if i ==5:
#         continue
#     print(i)
import pickle

with open(r"G:\PyCharm\Projectspro\image_caption\data\feature_extraction_inception_v3\image_features-0.pickle","rb") as f:
    a,b = pickle.load(f,encoding="latin")
    print(len(a),b.shape)
    # 100 (100, 1, 1, 2048)