# encoding: utf-8
"""实现简单KNN"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_data(file_path):
    data = pd.read_table(file_path, sep='\s+', header=None)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


class simple_KNN(object):
    def __init__(self, train_X, train_y, test_X, K = 3):
        """训练数据、标注以及测试数据"""
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.K = K

    def calculate_distance(self, test_x, train_X, train_y):
        """返回字典, test_x只能一行"""
        row_X = train_X.shape[0]
        test_data = np.tile(test_x, (row_X, 1))
        eucl_distance = np.sum((test_data - train_X)**2, axis=1)
        res = dict(zip(eucl_distance, train_y))
        return res

    def classify(self, test_data, train_data, label, verbose=False):
        """返回最大投票法选择的结果"""
        K = self.K
        res = self.calculate_distance(test_data, train_data, label)
        sorted_result = sorted(res.items(), key=lambda item: item[0], reverse=False)[0:K]
        if verbose:
            print(sorted_result)
        temp = [x[1] for x in sorted_result]
        output = max(temp, key=temp.count)
        if verbose:
            print("output is: ", output)
        return output

    def main(self, verbose=False):
        """处理全部数据"""
        test_row = self.test_X.shape[0]
        output = []
        for i in range(test_row):
            temp = self.classify(self.test_X[i], self.train_X, self.train_y)
            output.append(temp)
            # if verbose:
            #     print("test_X is: ", self.test_X[i],"result is: ", temp)
        if verbose:
            print("output is: ", output)
        return output

    def show_result(self, output, test_y):
        """展示结果"""
        test_y = list(test_y)
        plt.scatter(range(len(test_y)), test_y, marker='x', color='m', label='1', s=30)
        plt.scatter(range(len(test_y)), output, marker='+', color='c', label='2', s=50)
        p = 0
        for i in range(len(output)):
            if output[i] == test_y[i]:
                p += 1
        p = p/len(output)
        plt.title("accuracy: %s" % p)
        plt.show()




if __name__ == '__main__':
    train_X, test_X, train_y, test_y = get_data('date_test_data.txt')
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_y = list(train_y)
    knn = simple_KNN(train_X, train_y, test_X)
    output = knn.main(verbose=True)
    knn.show_result(output, test_y)