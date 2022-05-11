# -*- coding: utf-8 -*-
"""
t_test
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel


def main():
    coreg_path = "D:\\study\\multi_cotraining\\comparison\\coreg\\rmse"
    safer_path = "D:\\study\\multi_cotraining\\comparison\\safer\\log"
    mssra_path = "D:\\study\\multi_cotraining\\comparison\\mssra\\weka_result_after"
    bhd_path = "D:\\study\\multi_cotraining\\comparison\\bhd\\log_ratio\\20"
    samco_path = "D:\\study\\multi_cotraining\\log\\SamCo_xx\\SamCo_final\\20result"


    filename = ['Cloud', 'Hungary chickenpox', 'Stock', 'Laser', 'Concrete compressive strength', 'Airfoil self-noise',
            'space_ga', 'pollen', 'abalone', 'parkinsons', 'wind', 'wine_quality', 'bank8FM', 'cpu_small', 'kin8nm',
            'puma8NH', 'SeoulBikeData', 'elevators', 'AirQuality', 'Electrical_Grid_Stability']
    # filename = ['Cloud', 'Hungary chickenpox', 'Stock', 'Laser']
    cnt1 = ['0.1', '0.2', '0.3', '0.4']
    algorithm_name = np.array(['Datasets','COREG','SAFER','MSSRA','BHD']).reshape(1,-1)
    dataset_name = np.array(filename).reshape(-1,1)
    for j in range(4):
        t_test = np.zeros((len(filename), 4))
        for i in range(len(filename)):
            # print(filename[i])
            datapath1 = coreg_path + '/' + filename[i] + '_rmse_' + cnt1[j] + '.csv'
            datapath2 = safer_path + '/' + filename[i] + cnt1[j] + '.csv'
            datapath3 = mssra_path + '/' + filename[i] + cnt1[j] + '.csv'
            datapath4 = bhd_path + '/' + filename[i] + '_log_' + cnt1[j] + '.csv'
            datapath5 = samco_path + '/' + filename[i] + '_20_' + cnt1[j] + '_validate.csv'

            dt1 = pd.read_csv(datapath1, header=None)
            dt2 = pd.read_csv(datapath2, header=None)
            dt3 = pd.read_csv(datapath3, header=None)
            dt4 = pd.read_csv(datapath4)
            dt5 = pd.read_csv(datapath5, header=None)
            coreg_rmsefeature = np.array(dt1.iloc[0:dt1.shape[0], 0]).reshape(1, -1)
            safer_rmsefeature = np.array(dt2.iloc[0:dt2.shape[0], 1]).reshape(1, -1)
            mssra_rmsefeature = np.array(dt3.iloc[0:dt3.shape[0], 0]).reshape(1, -1)
            bhd_rmsefeature = np.array(dt4.iloc[0:dt4.shape[0], 0]).reshape(1, -1)
            samco_rmsefeature = np.array(dt5.iloc[0:dt5.shape[0], 0]).reshape(1, -1)

            _, t_test[i][0] = ttest_rel(coreg_rmsefeature[0, :].tolist(), samco_rmsefeature[0, :].tolist())
            _, t_test[i][1] = ttest_rel(safer_rmsefeature[0, :].tolist(), samco_rmsefeature[0, :].tolist())
            _, t_test[i][2] = ttest_rel(mssra_rmsefeature[0, :].tolist(), samco_rmsefeature[0, :].tolist())
            _, t_test[i][3] = ttest_rel(bhd_rmsefeature[0, :].tolist(), samco_rmsefeature[0, :].tolist())

        t_test_result = np.concatenate((dataset_name,t_test),axis=1)
        t_test_result = np.concatenate((algorithm_name,t_test_result),axis=0)

        result_dt = pd.DataFrame(t_test_result)
        result_dt.to_csv(os.getcwd() + '\\experiment\\t-test' + '\\' + 't_test_' + cnt1[j] + '.csv', index=False,header=False)



if __name__ == '__main__':
    main()
