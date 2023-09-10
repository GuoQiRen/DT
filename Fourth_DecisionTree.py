from collections import Counter
from math import log
import operator


def create_dataset():
    """ 创造示例数据 """
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍缩', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍缩', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍缩', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍缩', '沉闷', '稍糊', '稍凹', '硬滑', '好瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '硬滑', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '稍缩', '浊响', '稍糊', '凹陷', '软粘', '坏瓜'],
        ['浅白', '稍缩', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍缩', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    attributes = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']  # 6个特征
    return dataSet, attributes


def calc_entropy(data_set):
    """计算数据的熵(entropy)=-P*log_{2}*P"""
    sample_num = len(data_set)  # 数据条数
    label_counts = Counter()

    for feat_vec in data_set:
        current_label = feat_vec[-1]  # 每行数据的最后一个字（类别）
        label_counts[current_label] += 1  # 统计有多少个类以及每个类的数量

    entropy_value = 0

    for key in label_counts:
        prob = float(label_counts[key]) / sample_num  # 计算单个类的熵值
        entropy_value -= prob * log(prob, 2)  # 累加每个类的熵值

    return entropy_value


def split_data_set(data_set, axis, value):
    """按某个特征分类后的数据"""
    remain_data_set = []
    for featVec in data_set:
        if featVec[axis] == value:
            remain_feats = featVec[:axis]
            remain_feats.extend(featVec[axis + 1:])
            remain_data_set.append(remain_feats)
    return remain_data_set


def choose_best_feature_to_split(data_set, y_labels):
    """选择最优的分类特征"""
    num_features = len(data_set[0]) - 1
    base_entropy = calc_entropy(data_set)  # 当前根结点的熵值

    best_info_gain = 0
    best_feature = -1
    attributes = dict()

    for i in range(num_features):

        feat_list = [example[i] for example in data_set]
        unique_values = set(feat_list)
        new_entropy = 0

        # 计算每个特征属性值的熵总和
        for value in unique_values:

            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))

            # 计算label熵值
            new_entropy += prob * calc_entropy(sub_data_set)  # 按特征分类后的熵

        info_gain = base_entropy - new_entropy  # 原始熵与按特征分类后的熵的差值
        attributes[y_labels[i]] = info_gain

        if info_gain > best_info_gain:  # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_cnt(class_list):  # 按分类后类别数量排序，比如：最后分类为2好瓜1坏瓜，则判定为好瓜；
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, y_labels):
    class_list = [example[-1] for example in data_set]  # 类别：好瓜或坏瓜

    if class_list.count(class_list[0]) == len(class_list):  # 纯叶子结点，没有杂质
        return class_list[0]

    if len(data_set[0]) == 1:  # 仅剩一个特征属性，截止判断好坏结点数量即可
        return majority_cnt(class_list)

    best_feat_index = choose_best_feature_to_split(data_set, y_labels)  # 选择最优特征
    best_feat_label = y_labels[best_feat_index]

    my_tree = {best_feat_label: {}}  # 分类结果以字典形式保存
    del (y_labels[best_feat_index])

    feat_values = [example[best_feat_index] for example in data_set]
    unique_values = set(feat_values)

    for value in unique_values:
        sub_labels = y_labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat_index, value), sub_labels)

    return my_tree


if __name__ == '__main__':
    dataSet, attributes = create_dataset()
    tree = create_tree(dataSet, attributes)
    print(tree)
