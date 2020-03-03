# 由于频谱图生成含有噪声的判断的，某些音频被整个判断为噪声而不存在文件夹，因此需要进行校正，删除这些不存在的文件
import pandas as pd
import os

if __name__ == '__main__':
    # df = pd.read_csv(r'G:\dataset\BirdClef\vacation\source.csv')
    df = pd.read_csv(r'G:\dataset\BirdClef\vacation\target.csv')

    #spec_path = r'G:\dataset\BirdClef\vacation\spectrum\source'
    spec_path = r'G:\dataset\BirdClef\vacation\spectrum\target'

    pop_index = []
    for index, item in enumerate(df.iterrows()):
        item = item[1]
        filename = item['FileName']
        if not os.path.exists(os.path.join(spec_path, filename.split('.')[0])):
            print('{} dose not exist'.format(filename))
            pop_index.append(index)
    new_df = df.drop(pop_index)

    #new_df.to_csv(r'G:\dataset\BirdClef\vacation\source_check.csv')
    new_df.to_csv(r'G:\dataset\BirdClef\vacation\target_check.csv')