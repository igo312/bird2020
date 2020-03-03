import pandas as pd

df = [1998,1999,2000,2001]
df = pd.DataFrame(df, columns=['data1'])

# 原始转换 1970-01-01 00:00:00.000001998
a = pd.to_datetime(df['data1'])

# format属性确定格式 format='%Y' 因为提供的只有 年 的数据， 若有年月日则为'%y%m%d'
# 结果 1998-01-01
b = pd.to_datetime(df['data1'], format='%Y')