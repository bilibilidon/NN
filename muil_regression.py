import numpy as np
import pandas as pd
import statsmodels.api as sm

file = r'C:\Users\data.xlsx'
data = pd.read_excel(file)
# data.columns = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
# x = sm.add_constant(data.iloc[:,1:]) #生成自变量
# y = data['y'] #生成因变量
# model = sm.OLS(y, x) #生成模型
# result = model.fit() #模型拟合
# result.summary() #模型描述


def looper(limit):
    cols = ['x1', 'x2', 'x3', 'x5', 'x6', 'x7', 'x8', 'x9']
    for i in range(len(cols)):
        data1 = data[cols]
        x = sm.add_constant(data1)  # 生成自变量
        y = data['y']  # 生成因变量
        model = sm.OLS(y, x)  # 生成模型
        result = model.fit()  # 模型拟合
        pvalues = result.pvalues  # 得到结果中所有P值
        pvalues.drop('const', inplace=True)  # 把const取得
        pmax = max(pvalues)  # 选出最大的P值
        if pmax > limit:
            ind = pvalues.idxmax()  # 找出最大P值的index
            cols.remove(ind)  # 把这个index从cols中删除
        else:
            return result


result = looper(0.05)
result.summary()

# import matplotlib.pyplot as plt
# y_pred = model.predict(X)
# fig,ax = plt.subplots(figsize=(8,4))
# ax.plot(Y,color='#00b0ff',label="Observations",linewidth=1.5)
# ax.plot(y_pred,color='#ff3d00',label="Prediction",linewidth=1.5)
# ax.legend(loc="upper left",fontsize=12)
# ax.grid(alpha=0.6)
# ax.tick_params(labelsize=14)
