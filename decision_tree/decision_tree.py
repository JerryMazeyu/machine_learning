import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DecisionTree(object):
    def __init__(self, file_path, mapping_list=None):
        """属性有文件路径与映射表"""
        self.file_path = file_path
        self.mapping_list = mapping_list


    def pre_process(self, verbose=False):
        """数据预处理，包含读取数据与分类"""
        file_path = self.file_path
        mapping_list = self.mapping_list
        data = pd.read_table(file_path, sep='\s+')
        if mapping_list:
            for i in range(len(mapping_list)):
                dict = mapping_list[i]
                key = list(dict.keys())[0]
                value = list(dict.values())[0]
                data[key] = data[key].map(value)
        if verbose:
            print(data)
        return data


    def _split_data(self, verbose=False):
        """数据分割"""
        data = self.pre_process()
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
        if verbose:
            print(X_train, "\n", X_test, "\n", y_train, "\n", y_test)
        return X_train, X_test, y_train, y_test


    def _calculate_entropy(self, y):
        """计算HD，y是列表形式"""
        set_y = set(y)
        dict01 = {}
        for item in set_y:
            dict01.update({item: y.count(item) / len(y)})
        p_y = np.array([list(dict01.values())])
        H_D = -np.sum(p_y * np.log(p_y) / np.log(2), axis=1)
        return H_D[0]


    def _calculate_gain(self, X, y, D, verbose=False):
        """计算信息增益"""
        H_D = self._calculate_entropy(list(y))
        data = pd.concat([X, y], axis=1)
        set_D = list(set(X[D]))
        H_D_A = 0
        for i in set_D:
            temp = data.loc[data[D] == i]
            p = len(temp) / len(X)
            res = self._calculate_entropy(list(temp.iloc[:, -1]))
            H_D_A += res * p
        if verbose:
            print("D is %s, ID3 gain is %s" % (D, H_D - H_D_A))
        return H_D - H_D_A


    def _calculate_gain_ratio(self, X, y, D, verbose=False):
        """C4.5信息增益率"""
        gain = self._calculate_gain(X, y, D)
        name = D
        D = list(X[D])
        IV_D = self._calculate_entropy(D) + 1e-10
        if verbose:
            print("D is %s, C4.5 gain_ratio is %s: " % (name, gain/IV_D))
        return gain/IV_D


    def decision(self, X, y, kernel="C4.5", treshold=0.0, verbose=False):
        """决定过程"""
        node_name = list(X.columns)
        if kernel == 'C4.5':
            gain = {}
            for i in node_name:
                gain[i] = self._calculate_gain_ratio(X, y, i)
        elif kernel == 'ID3':
            gain = {}
            for i in node_name:
                gain[i] = self._calculate_gain(X, y, i)
        if verbose:
            print("gain is: ", gain)
        sorted_res = sorted(gain.items(), key=lambda item: item[1], reverse=True)
        temp = [x[0] for x in sorted_res]
        node_name = temp[0]
        if gain[node_name] < treshold:
            print("No")
            return [None], [None], [None]
        data = pd.concat([X, y], axis=1)
        output_X = []
        for i in data.iloc[:, :-1].groupby(data[node_name]):
            output_X.append(i[1])
        output_y = []
        for i in data.iloc[:, -1].groupby(data[node_name]):
            output_y.append(i[1])
        if verbose:
            # print("output_X is: ", "\n", output_y[0])
            print("node_name is: ", node_name)
        return output_X, output_y, node_name


    def _is_pure(self, y):
        """查看分类后的节点中是不是全是一个类别的"""
        if len(set(y)) == 1:
            return True
        else:
            return False


    def main(self, X, y, kernel='C4.5', verbose=False):
        """递归建立决策树"""
        data = pd.concat([X, y], axis=1)
        node_names = data.columns
        if verbose:
            print("node_names is: ", node_names)
        result = []
        queue = []
        queue.append([X, y])
        while queue:
            temp = {}
            X, y = queue.pop(0)
            output_X, output_y, nodename = self.decision(X, y, kernel=kernel, verbose=verbose)
            temp[nodename] = {}
            for i in range(len(output_y)):
                mazeyuhahaha = list(output_X[i][nodename])[0]
                if output_y[i] is None:
                    if verbose:
                        print("lower than treshold")
                    temp[nodename][mazeyuhahaha] = list(set(list(output_y[i])))[0]
                    continue
                if self._is_pure(output_y[i]):
                    if verbose:
                        print("i，nodename is: ", i, nodename, "\n", pd.concat([output_X[i], output_y[i]], axis=1))
                    temp[nodename][mazeyuhahaha] = list(set(list(output_y[i])))[0]
                    pass
                else:
                    temp[nodename][mazeyuhahaha] = [None, list(set(list(output_y[i])))[0]]
                    queue.append([output_X[i], output_y[i]])
            result.append(temp)
        print("result is: ", result)
        return result


    def predict(self, X_test, result):
        """用拟合的决策树进行预测"""
        y_hat = None
        for i in range(len(result)):
            nodename = list(result[i].keys())[0]
            cond = list(result[i].values())[0]
            for j in cond.keys():
                value = cond[j]
                if X_test[nodename].iloc[0] == j:
                    if isinstance(value, list):
                        print(nodename, value[1], '->', end='')
                        y_hat = value[1]
                    else:
                        print(nodename, value)
                        y_hat = value
        return y_hat


    def cut_tree(self, result, type='layer', layer=1):
        """指定层数的伪预剪枝"""
        if type == 'layer':
            return result[:layer]






if __name__ == '__main__':
    mapping_list = [{'年龄': {'青年': 0, "中年": 1, "老年": 2}}, {'有工作': {'是': 0, '否': 1}}, {"有自己的房子": {'是': 0, '否': 1}},
                    {'信贷情况': {'一般': 0, "好": 1, "非常好": 2}}, {'类别(是否个给贷款)': {'是': 'A', '否': 'B'}}]
    file_path = 'debt.txt'
    decision_tree = DecisionTree(file_path, mapping_list)
    file = decision_tree.pre_process(verbose=False)
    X_train, X_test, y_train, y_test = decision_tree._split_data(verbose=False)
    # D = '年龄'
    # decision_tree._calculate_gain(X_train, y_train, D, verbose=False)
    # decision_tree._calculate_gain_ratio(X_train, y_train, D, verbose=False)
    # output_X, output_y, node_name = decision_tree.decision(X_train, y_train, verbose=True)
    result = decision_tree.main(X_train, y_train, verbose=True)
    # print(X_test['有工作'].iloc[0])
    decision_tree.predict(X_test, result)
    cuted_tree = decision_tree.cut_tree(result)
    decision_tree.predict(X_test, cuted_tree)