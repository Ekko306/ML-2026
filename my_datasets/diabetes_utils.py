"""
糖尿病数据集处理工具模块
========================
把 my_datasets/diabetes.ipynb 中的数据处理流程包装成可复用函数：
    1 获取数据 → 数据清洗检查（diabetes 本身干净，仅保留防御性清理）→ 6 数据拆分 → 5 特征处理(标准化)

其他 notebook 中的用法：
    from diabetes_utils import load_diabetes_ready
    x_train, x_test, y_train, y_test, scaler = load_diabetes_ready()
"""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_diabetes_ready(test_size=0.2, random_state=22, scale=True):
    """一键获取"清理检查 + 拆分 + 标准化"后的糖尿病回归数据。

    参数:
        test_size (float):   测试集比例，默认 0.2（89 条测试 / 353 条训练）
        random_state (int):  随机种子，固定后每次拆分结果一致，保证实验可复现
        scale (bool):        是否用 StandardScaler 标准化特征，默认 True

    返回:
        x_train (DataFrame): 训练集特征 (353, 10)
        x_test  (DataFrame): 测试集特征 (89, 10)
        y_train (Series):    训练集目标值（连续的疾病进展评分 → 回归任务）
        y_test  (Series):    测试集目标值
        scaler  (object):    在训练集上 fit 好的 StandardScaler；
                             后续新数据请用它 transform，保持变换规则一致
                             （scale=False 时返回 None）
    """
    # ---------- 1 获取数据 ----------
    # 从 sklearn 加载糖尿病数据集，转成 DataFrame（特征已被 sklearn 中心化并按列归一化）
    diabetes = load_diabetes()
    data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    data['target'] = diabetes.target

    # ---------- 数据清洗（防御性检查） ----------
    # notebook 分析结论：diabetes 自带数据无缺失、无重复、无异常录入，本可跳过；
    # 但真实项目里数据源不可控，保留这三步防御性清理，成本极低
    data = data.dropna()           # 若有缺失值所在的行则删除
    data = data.drop_duplicates()  # 若有完全重复的行则删除

    # 分离特征 X（前 10 列）和标签 y（target 列，连续值 → 回归任务）
    X = data[diabetes.feature_names]
    y = data['target']

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
    # 直接运行本文件可自测：python diabetes_utils.py
    x_train, x_test, y_train, y_test, scaler = load_diabetes_ready()
    print("x_train:", x_train.shape)   # 预期 (353, 10)
    print("x_test :", x_test.shape)    # 预期 (89, 10)
    print("y_train 取值范围: [%d, %d]" % (y_train.min(), y_train.max()))
    print("标准化后训练集均值(应≈0):", x_train.mean().abs().max().round(6))
    print("标准化后训练集标准差(应≈1):", x_train.std().round(6).min(), "~", x_train.std().round(6).max())
