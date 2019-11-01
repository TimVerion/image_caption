import os
import sys
import tensorflow as tf
from tensorflow import gfile  # 负责读取pb文件,构造图
from tensorflow import logging
# import pprint
import pickle
import numpy as np

# 使用inception v3 来对图像进行提取特征
# 对于checkpoint存的各个变量的数值,使用restore加载
# 除了上述的,还可以使用freeze_graph将生成的模型合并成一个pb文件
# 这样使用就不用重构原来的图
model_file = "./data/classify_image_graph_def.pb"
input_description_file = "./data/results_20130124.token"
input_img_dir = "./data/flickr30k-images/"
output_folder = "./data/feature_extraction_inception_v3"

batch_size = 100

if not gfile.Exists(output_folder):
    gfile.MakeDirs(output_folder)


def parse_token_file(token_file):
    """解析token文件"""
    img_name_to_tokens = {}
    with gfile.GFile(token_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        img_id, description = line.strip('\r\n').split('\t')
        img_name, _ = img_id.split('#')
        img_name_to_tokens.setdefault(img_name, [])
        img_name_to_tokens[img_name].append(description)
    return img_name_to_tokens


img_name_to_tokens = parse_token_file(input_description_file)
all_img_names = list(img_name_to_tokens.keys())


# logging.info("num of all images: %d" % len(all_img_names))
# pprint.pprint(img_name_to_tokens.keys()[0:10])
# pprint.pprint(img_name_to_tokens['2778832101.jpg'])

def load_pretrained_inception_v3(model_file):
    with gfile.FastGFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()  # 构造一个空的图
        graph_def.ParseFromString(f.read())  # 将计算图读取进来
        _ = tf.import_graph_def(graph_def, name="")  # 将图导入到默认图


load_pretrained_inception_v3(model_file)

num_batches = int(len(all_img_names) / batch_size)  # 计算每批次batch_size个共有多少个子文件
if len(all_img_names) % batch_size != 0:
    # 将多余的文件使用一个batch
    num_batches += 1

with tf.Session() as sess:
    second_to_last_tensor = sess.graph.get_tensor_by_name("pool_3:0")  # 获得输出tensor
    for i in range(num_batches):
        batch_img_names = all_img_names[i * batch_size: (i + 1) * batch_size]  # 按批量去除name
        batch_features = []
        for img_name in batch_img_names:
            img_path = os.path.join(input_img_dir, img_name)
            logging.info("processing img %s" % img_name)
            if not gfile.Exists(img_path):
                raise Exception("%s doesn't exists" % img_path)
            img_data = gfile.FastGFile(img_path, "rb").read()  # 读取图片
            feature_vector = sess.run(second_to_last_tensor,
                                      feed_dict={"DecodeJpeg/contents:0": img_data})
            batch_features.append(feature_vector)
        batch_features = np.vstack(batch_features)
        output_filename = os.path.join(output_folder, "image_features-%d.pickle" % i)
        # logging.info("writing to file %s" % output_filename)
        with gfile.GFile(output_filename, 'w') as f:  # 打开一个文件
            pickle.dump((batch_img_names, batch_features), f)  # 将数据保存在文件中
        # 图像预处理完成
