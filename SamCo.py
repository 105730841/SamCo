'''
rfr,knn,rr,mlp
train:validate:test=5:1:4
'''

import torch
import os
import random as rd
import numpy as np
import cvxopt
import pandas as pd
import copy
import torch.nn.functional as F  # 激励函数都在这
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neighbors import KNeighborsRegressor as Knn
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
import math
import time
import joblib


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1234)


# mlp
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden1, n_hidden2,
                 n_output):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_feature + n_hidden2, n_output)

    def forward(self, x):
        inputs = x
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = torch.cat([inputs, x], dim=-1)
        x = self.predict(x)
        return x


def main():
    path1 = os.getcwd() + '\data'
    # data = ['cpu_small', 'Folds5x2_pp', 'bank8FM', 'puma8NH', 'kin8nm', 'parkinsons', 'wind', 'wine_quality',
    #         'space_ga', 'abalone', 'Electrical_Grid_Stability', 'elevators', 'SeoulBikeData']
    data = ['Cloud', 'Hungary chickenpox', 'Stock', 'Laser', 'Concrete compressive strength', 'Airfoil self-noise',
            'space_ga', 'pollen', 'abalone', 'parkinsons', 'wind', 'wine_quality', 'bank8FM', 'cpu_small', 'kin8nm',
            'puma8NH', 'SeoulBikeData', 'elevators', 'AirQuality', 'Electrical_Grid_Stability']
    for i in range(len(data)):
        dataname = data[i]
        if i > 4:
            lr = 0.01
            train_lr = 0.1
        else:
            lr = 0.001
            train_lr = 0.1
        result = np.array(
            ['ratio', 'semi_SamCo', 'r2']).reshape(1, -1)
        for j in range(4):
            ratio = round(0.1 * (j + 1), 1)
            file_number = 20

            semi_samco = np.zeros(file_number)
            r2_result = np.zeros(file_number)

            for k in range(file_number):
                print(dataname, k, ratio)
                path = path1 + '\\%s' % (dataname) + '.arff_{}.csv'.format(k)
                f = open(path, encoding='utf-8')
                train_data, train_label, unlabel_data, unlabel_label, validate_data, validate_label, test_data, test_label, num_features = read_data(
                    f, label_ratio=ratio, train_ratio=0.5, validate_ratio=0.1)

                # Initialization
                learn_rfr = RFR()
                train_label_ravel = np.ravel(train_label)
                learn_rfr.fit(train_data, train_label_ravel)

                learn_knn = Knn(n_neighbors=3, weights='uniform')
                learn_knn.fit(train_data, train_label)

                learn_rr = Ridge(alpha=0.5)
                learn_rr.fit(train_data, train_label)

                learn_mlp = Net(n_feature=num_features, n_hidden=32, n_hidden1=64, n_hidden2=32, n_output=1)
                learn_mlp = train_iteration(learn_mlp, train_data, train_label, 3000, lr)

                # Train
                rfr, knn, rr, mlp = samco(learn_rfr, learn_knn, learn_rr, learn_mlp, train_data, train_label,
                                                   unlabel_data, test_data, test_label, 100, train_lr)

                # Calculating the weight of weak regressors
                rfr_validate_pre = rfr.predict(validate_data).reshape(-1, 1)
                knn_validate_pre = knn.predict(validate_data).reshape(-1, 1)
                rr_validate_pre = rr.predict(validate_data).reshape(-1, 1)
                validate_data_torch = torch.from_numpy(validate_data)
                mlp_validate_pre = mlp(validate_data_torch).data.numpy()

                mse_rfr = mse(validate_label, rfr_validate_pre)
                mse_knn = mse(validate_label, knn_validate_pre)
                mse_rr = mse(validate_label, rr_validate_pre)
                mse_mlp = mse(validate_label, mlp_validate_pre)

                weight_rfr = pow((1 / mse_rfr), 3) / (
                        pow((1 / mse_rfr), 3) + pow((1 / mse_knn), 3) + pow((1 / mse_rr), 3) + pow((1 / mse_mlp),
                                                                                                   3))
                weight_knn = pow((1 / mse_knn), 3) / (
                        pow((1 / mse_rfr), 3) + pow((1 / mse_knn), 3) + pow((1 / mse_rr), 3) + pow((1 / mse_mlp),
                                                                                                   3))
                weight_rr = pow((1 / mse_rr), 3) / (
                        pow((1 / mse_rfr), 3) + pow((1 / mse_knn), 3) + pow((1 / mse_rr), 3) + pow((1 / mse_mlp),
                                                                                                   3))
                weight_mlp = pow((1 / mse_mlp), 3) / (
                        pow((1 / mse_rfr), 3) + pow((1 / mse_knn), 3) + pow((1 / mse_rr), 3) + pow((1 / mse_mlp),
                                                                                                   3))
                # Prediction
                pre_rfr = rfr.predict(test_data).reshape(-1, 1)
                pre_knn = knn.predict(test_data).reshape(-1, 1)
                pre_rr = rr.predict(test_data).reshape(-1, 1)
                test_data_torch = torch.from_numpy(test_data)
                pre_mlp = mlp(test_data_torch).data.numpy()
                pre_final = pre_knn * weight_knn + pre_rfr * weight_rfr + pre_rr * weight_rr + pre_mlp * weight_mlp

                # Store the result
                semi_samco[k] = math.sqrt(mse(test_label, pre_final))
                r2_result[k] = R_2(test_label, pre_final)  # R2

            rmse_result = semi_samco.reshape(-1, 1)
            rmse_20_dt = pd.DataFrame(rmse_result)
            rmse_20_name = dataname + '_20_{}'.format(ratio)
            rmse_20_dt.to_csv(os.getcwd() + '\\log\\SamCo_final\\result' + '\\' + rmse_20_name + '.csv', index=False, header=False)

            temp_result = np.array([ratio, semi_samco.mean(), r2_result.mean()]).reshape(1, -1)
            result = np.concatenate((result, temp_result), axis=0)
        result_dt = pd.DataFrame(result)
        result_dt.to_csv(os.getcwd() + '\\log\\SamCo_final\\total' + '\\' + dataname + '_total.csv', index=False, header=False)



def samco(learn_rfr, learn_knn, learn_rr, learn_mlp, train_data, train_label, unlabel_data, test_data, test_label,
          iter_max, lr):
    rfr = copy.deepcopy(learn_rfr)
    knn = copy.deepcopy(learn_knn)
    rr = copy.deepcopy(learn_rr)
    mlp = copy.deepcopy(learn_mlp)

    temp_train_data = copy.deepcopy(train_data)
    temp_train_label = copy.deepcopy(train_label)

    temp_add_data_rfr = np.empty(shape=[0, train_data.shape[1]])
    temp_add_data_knn = np.empty(shape=[0, train_data.shape[1]])
    temp_add_data_rr = np.empty(shape=[0, train_data.shape[1]])
    temp_add_data_mlp = np.empty(shape=[0, train_data.shape[1]])

    for i in range(iter_max):
        print(i)
        if len(unlabel_data) == 0:
            break
        # 1 For rfr
        # Selection
        unlabel_data_mlp = torch.from_numpy(unlabel_data)
        temp_mlp_pre = mlp(unlabel_data_mlp).data.numpy()
        temp_knn_pre = knn.predict(unlabel_data).reshape(-1, 1)
        temp_rr_pre = rr.predict(unlabel_data).reshape(-1, 1)
        temp_rfr = np.concatenate((temp_mlp_pre, temp_knn_pre, temp_rr_pre), axis=1)
        var_rfr = temp_rfr.var(axis=1)
        sort_var_rfr = np.argsort(var_rfr)
        idx_rfr = sort_var_rfr[0]
        confident_instance_rfr = unlabel_data[idx_rfr].reshape(1, -1)

        # pseudo-label
        temp_add_data_rfr = np.concatenate((temp_add_data_rfr, confident_instance_rfr), axis=0)
        mlp_add_data_rfr = torch.from_numpy(temp_add_data_rfr).float()
        mlp_pre_rfr = mlp(mlp_add_data_rfr).data.numpy()
        knn_pre_rfr = knn.predict(temp_add_data_rfr).reshape(-1, 1)
        rr_pre_rfr = rr.predict(temp_add_data_rfr).reshape(-1, 1)
        rfr_pre = rfr.predict(temp_add_data_rfr).reshape(-1, 1)
        convex_rfr = np.concatenate((mlp_pre_rfr, knn_pre_rfr, rr_pre_rfr), axis=1)
        try:
            temp_pre_rfr = safe_labeling(convex_rfr, rfr_pre)
        except:
            temp_pre_rfr = mlp_pre_rfr

        temp_train_data_rfr = np.concatenate((temp_train_data, temp_add_data_rfr))
        temp_train_label_rfr = np.concatenate((temp_train_label, temp_pre_rfr))
        temp_train_label_rfr_ravel = np.ravel(temp_train_label_rfr)
        rfr.fit(temp_train_data_rfr, temp_train_label_rfr_ravel)

        unlabel_data = np.delete(unlabel_data, idx_rfr, axis=0)

        if len(unlabel_data) == 0:
            break
        # 2 For knn
        unlabel_data_mlp = torch.from_numpy(unlabel_data)
        temp_mlp_pre = mlp(unlabel_data_mlp).data.numpy()
        temp_rfr_pre = rfr.predict(unlabel_data).reshape(-1, 1)
        temp_rr_pre = rr.predict(unlabel_data).reshape(-1, 1)
        temp_knn = np.concatenate((temp_mlp_pre, temp_rfr_pre, temp_rr_pre), axis=1)
        var_knn = temp_knn.var(axis=1)
        sort_var_knn = np.argsort(var_knn)
        idx_knn = sort_var_knn[0]
        confident_instance_knn = unlabel_data[idx_knn].reshape(1, -1)

        temp_add_data_knn = np.concatenate((temp_add_data_knn, confident_instance_knn), axis=0)
        mlp_add_data_knn = torch.from_numpy(temp_add_data_knn).float()
        mlp_pre_knn = mlp(mlp_add_data_knn).data.numpy()
        rfr_pre_knn = rfr.predict(temp_add_data_knn).reshape(-1, 1)
        rr_pre_knn = rr.predict(temp_add_data_knn).reshape(-1, 1)
        knn_pre = knn.predict(temp_add_data_knn).reshape(-1, 1)
        convex_knn = np.concatenate((mlp_pre_knn, rfr_pre_knn, rr_pre_knn), axis=1)
        try:
            temp_pre_knn = safe_labeling(convex_knn, knn_pre)
        except:
            temp_pre_knn = rfr_pre_knn

        temp_train_data_knn = np.concatenate((temp_train_data, temp_add_data_knn))
        temp_train_label_knn = np.concatenate((temp_train_label, temp_pre_knn))
        knn.fit(temp_train_data_knn, temp_train_label_knn)

        unlabel_data = np.delete(unlabel_data, idx_knn, axis=0)

        if len(unlabel_data) == 0:
            break
        # 3 For rr
        unlabel_data_mlp = torch.from_numpy(unlabel_data)
        temp_mlp_pre = mlp(unlabel_data_mlp).data.numpy()
        temp_rfr_pre = rfr.predict(unlabel_data).reshape(-1, 1)
        temp_knn_pre = knn.predict(unlabel_data).reshape(-1, 1)
        temp_rr = np.concatenate((temp_mlp_pre, temp_rfr_pre, temp_knn_pre), axis=1)
        var_rr = temp_rr.var(axis=1)
        sort_var_rr = np.argsort(var_rr)
        idx_rr = sort_var_rr[0]
        confident_instance_rr = unlabel_data[idx_rr].reshape(1, -1)

        temp_add_data_rr = np.concatenate((temp_add_data_rr, confident_instance_rr), axis=0)
        mlp_add_data_rr = torch.from_numpy(temp_add_data_rr).float()
        mlp_pre_rr = mlp(mlp_add_data_rr).data.numpy()
        rfr_pre_rr = rfr.predict(temp_add_data_rr).reshape(-1, 1)
        knn_pre_rr = knn.predict(temp_add_data_rr).reshape(-1, 1)
        rr_pre = rr.predict(temp_add_data_rr).reshape(-1, 1)
        convex_rr = np.concatenate((mlp_pre_rr, rfr_pre_rr, knn_pre_rr), axis=1)

        try:
            temp_pre_rr = safe_labeling(convex_rr, rr_pre)
        except:
            temp_pre_rr = rfr_pre_rr

        temp_train_data_rr = np.concatenate((temp_train_data, temp_add_data_rr))
        temp_train_label_rr = np.concatenate((temp_train_label, temp_pre_rr))
        temp_train_label_rr_ravel = np.ravel(temp_train_label_rr)
        rr.fit(temp_train_data_rr, temp_train_label_rr_ravel)

        unlabel_data = np.delete(unlabel_data, idx_rr, axis=0)

        if len(unlabel_data) == 0:
            break
        # 4 For mlp
        temp_rfr_pre = rfr.predict(unlabel_data).reshape(-1, 1)
        temp_knn_pre = knn.predict(unlabel_data).reshape(-1, 1)
        temp_rr_pre = rr.predict(unlabel_data).reshape(-1, 1)
        temp_mlp = np.concatenate((temp_rfr_pre, temp_knn_pre, temp_rr_pre), axis=1)
        var_mlp = temp_mlp.var(axis=1)
        sort_var_mlp = np.argsort(var_mlp)
        idx_mlp = sort_var_mlp[0]
        confident_instance_mlp = unlabel_data[idx_mlp].reshape(1, -1)

        temp_add_data_mlp = np.concatenate((temp_add_data_mlp, confident_instance_mlp), axis=0)
        mlp_add_data_mlp = torch.from_numpy(temp_add_data_mlp).float()
        rfr_pre_mlp = rfr.predict(temp_add_data_mlp).reshape(-1, 1)
        knn_pre_mlp = knn.predict(temp_add_data_mlp).reshape(-1, 1)
        rr_pre_mlp = rr.predict(temp_add_data_mlp).reshape(-1, 1)
        mlp_pre = mlp(mlp_add_data_mlp).data.numpy()
        convex_mlp = np.concatenate((rfr_pre_mlp, knn_pre_mlp, rr_pre_mlp), axis=1)
        try:
            temp_pre_mlp = safe_labeling(convex_mlp, mlp_pre)
        except:
            temp_pre_mlp = rfr_pre_mlp

        temp_train_data_mlp = np.concatenate((temp_train_data, temp_add_data_mlp))
        temp_train_label_mlp = np.concatenate((temp_train_label, temp_pre_mlp))
        mlp = train_iteration(mlp, temp_train_data_mlp, temp_train_label_mlp, 100, lr)

        unlabel_data = np.delete(unlabel_data, idx_mlp, axis=0)
    return rfr, knn, rr, mlp


def read_data(path, label_ratio, train_ratio, validate_ratio):
    data = pd.read_csv(path)
    all_features = data.iloc[:, 0:data.shape[1] - 1]
    all_labels = data.iloc[:, data.shape[1] - 1:data.shape[1]]
    all_features = all_features.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    all_labels = all_labels.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    num_index = all_features.shape[0]
    num_features = all_features.shape[1]
    data = all_features[0:num_index].values.astype(np.float32)
    label = all_labels[0:num_index].values.astype(np.float32)
    label_Index = round(num_index * train_ratio * label_ratio)
    unlabel_Index = round(num_index * train_ratio)
    validate_Index = round(num_index * (train_ratio + validate_ratio))
    # label data
    train_data = data[0:label_Index, :]
    train_label = label[0:label_Index, :]
    # unlabel data
    unlabel_data = data[label_Index:unlabel_Index, :]
    unlabel_label = label[label_Index:unlabel_Index, :]
    # validate data
    validate_data = data[unlabel_Index:validate_Index, :]
    validate_label = label[unlabel_Index:validate_Index, :]
    test_data = data[validate_Index:data.shape[0], :]
    test_label = label[validate_Index:label.shape[0], :]
    return train_data, train_label, unlabel_data, unlabel_label, validate_data, validate_label, test_data, test_label, num_features


def R_2(label, predict_label):
    r_2 = 0
    label = label.reshape(-1, 1)
    predict_label = predict_label.reshape(-1, 1)
    # 定义字典
    result = {}
    ybar = np.sum(label) / label.shape[0]
    ssreg = np.sum((predict_label - label) ** 2)
    sstot = np.sum((label - ybar) ** 2)
    r_2 = 1 - ssreg / sstot
    result['R^2'] = r_2
    return r_2


def train_iteration(model, data, label, num_epoch, learning_rate):
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).float()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 传入 net 的所有参数, 学习率, 使用随机梯度下降优化
    loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

    for e in range(num_epoch):
        label_pre = model(data)
        loss = loss_func(label_pre, label)
        if (e % 100 == 0):
            print(f'Epoch:{e},loss:{loss.data.numpy()}')  # !!!!!!!!!!
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
    return model


def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert (k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert (Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq, lb, ub)

    return np.array(sol['x'])


def safe_labeling(candidate_prediction, baseline_prediction):
    semi_pre = copy.deepcopy(candidate_prediction.astype(np.float64))
    supervised_pre = copy.deepcopy(baseline_prediction)
    prediction_num = candidate_prediction.shape[1]
    H = np.dot(semi_pre.T, semi_pre) * 2
    f = -2 * np.dot(semi_pre.T, supervised_pre)
    Aeq = np.ones((1, prediction_num))
    beq = 1.0
    lb = np.zeros((prediction_num, 1))
    ub = np.ones((prediction_num, 1))
    sln = quadprog(H, f, None, None, Aeq, beq, )
    safer_prediction = np.zeros((semi_pre.shape[0], 1))
    for i in range(safer_prediction.shape[0]):
        tempsafer = 0
        for j in range(prediction_num):
            tempsafer = tempsafer + sln[j] * semi_pre[i, j]
        safer_prediction[i][0] = tempsafer
    return safer_prediction


if __name__ == '__main__':
    main()
