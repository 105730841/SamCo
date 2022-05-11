import Orange
# import orngStat
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Adam-LDL-SCL x^2
names = ['COREG', 'SAFER', 'MSSRA', 'BHD', 'SamCo']
avranks10 = [3.15, 3.15, 2.5, 4.8, 1.4]
avranks20 = [3.05, 3.3, 2.45, 4.8, 1.4]
avranks30 = [3, 3.45, 2.35, 4.8, 1.4]
avranks40 = [2.95, 3.45, 2.4, 4.8, 1.4]


 #tested on 25 datasets
cd = Orange.evaluation.compute_CD(avranks40, 20, alpha='0.05', test='bonferroni-dunn')
print('cd=', cd)
Orange.evaluation.graph_ranks(avranks40, names, cd=cd, width=8, reverse=True)
plt.savefig('D:\study\multi_cotraining\experiment\cd\cd0.4.pdf', format='pdf')
plt.show()