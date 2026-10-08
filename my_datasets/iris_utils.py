"""
鸢尾花数据集处理工具模块
========================
把 datasets/iris.ipynb 中 1-6 步数据处理流程包装成可复用函数：
    1 获取数据 → 4 数据清洗 → 5 特征处理(标准化) → 6 数据拆分
（第 2 步描述性统计、第 3 步可视化属于"分析"，不改变数据，因此不在函数内）

其他 notebook 中的用法：
    from iris_utils import load_iris_ready
    x_train, x_test, y_train, y_test, scaler = load_iris_ready()
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_iris_ready(test_size=0.2, random_state=22, scale=True):
    """一键获取"清洗 + 标准化 + 拆分"后的鸢尾花数据。

    参数:
        test_size (float):   测试集比例，默认 0.2（30 条测试 / 119 条训练）
        random_state (int):  随机种子，固定后每次拆分结果一致，保证实验可复现
        scale (bool):        是否用 StandardScaler 标准化特征，默认 True

    返回:
        x_train (DataFrame): 训练集特征 (119, 4)
        x_test  (DataFrame): 测试集特征 (30, 4)
        y_train (Series):    训练集标签（品种名字符串）
        y_test  (Series):    测试集标签
        scaler  (object):    在训练集上 fit 好的 StandardScaler；
                             后续新数据请用它 transform，保持变换规则一致
                             （scale=False 时返回 None）
    """
    # ---------- 1 获取数据 ----------
    # 从 sklearn 加载鸢尾花数据集，转成 DataFrame 并把数字标签 0/1/2 映射成品种名
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target_names[iris.target]

    # ---------- 4 数据清洗 ----------
    # 删除缺失值所在的行
    data = data.dropna()
    # 删除完全重复的行（iris 原始数据里有 1 条重复样本）
    data = data.drop_duplicates()
    # 异常值过滤：4 个特征都应落在 (0, 10] cm 的合理范围内
    for col in iris.feature_names:
        data = data[(data[col] > 0) & (data[col] <= 10)]
    # 删除行后索引出现"断号"，重置为 0~n 连续索引
    data = data.reset_index(drop=True)

    # 分离特征 X（前 4 列）和标签 y（species 列）
    X = data[iris.feature_names]
    y = data['species']

    # ---------- 6 数据拆分（先拆分，再标准化） ----------
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # ---------- 5 特征处理：标准化 ----------
    # 关键原则：fit 只能在训练集上做！
    # 训练集 fit_transform，测试集只 transform —— 否则测试集信息会"泄露"到训练过程，
    # 且两者的变换规则可能不一致，导致评估结果不可信
    if scale:
        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train),
                               columns=X.columns, index=x_train.index)
        x_test = pd.DataFrame(scaler.transform(x_test),
                              columns=X.columns, index=x_test.index)
    else:
        scaler = None

    return x_train, x_test, y_train, y_test, scaler


if __name__ == "__main__":
    # 直接运行本文件可自测：python iris_utils.py
    x_train, x_test, y_train, y_test, scaler = load_iris_ready()
    print("x_train:", x_train.shape)   # 预期 (119, 4)
    print("x_test :", x_test.shape)    # 预期 (30, 4)
    print("y_train 品种分布:\n", y_train.value_counts())
    print("标准化后训练集均值(应≈0):\n", x_train.mean().round(6))
    print("标准化后训练集标准差(应≈1):\n", x_train.std().round(6))
