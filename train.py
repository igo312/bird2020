import os
import pandas as pd

path1 = r'G:\dataset\BirdClef\vacation\ica\ica30'
path2 = r'G:\dataset\BirdClef\vacation\ica\ica10'

df = r'G:\dataset\BirdClef\vacation\lssource_limit30.csv'
df = pd.read_csv(df)
for name in df.FileName:
    name1 = 'ica30'+name+'.mat'
    name2 = name+'.mat'
    if os.path.exists(os.path.join(path1, name1)) or os.path.exists(os.path.join(path2, name2)):
        pass
    else:
        print('%s not exists' %name)