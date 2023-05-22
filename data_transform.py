import pandas as pd# 读取excel文件

data = pd.read_excel('G:/beifen/PycharmProjects/water_pretiction/west.xlsx')# 保存为csv文件
data.to_csv('G:/beifen/PycharmProjects/water_pretiction/west.csv', index=False)