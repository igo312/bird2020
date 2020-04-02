import shutil
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    # Split data into source and target label
    _dst = r'G:\dataset\BirdClef\vacation\spectrum'
    _src = r'G:\dataset\BirdClef\paper_dataset\spectrum'
    label_file = pd.read_excel(r'G:\dataset\BirdClef\vacation\label.xlsx')
    data_file = pd.read_csv(r'G:\dataset\BirdClef\paper_dataset\static.csv')

    #label = list(np.asarray(label_file).reshape([-1]))
    source = data_file[data_file.Species.isin(label_file.SOURCE_NAME)]
    target = data_file[data_file.Species.isin(label_file.TARGET_NAME)]
    source.to_csv(r'G:\dataset\BirdClef\vacation\source.csv')
    target.to_csv(r'G:\dataset\BirdClef\vacation\target.csv')
    print('Label Record Done')

    '''
    # xml data moving part
    _dst = r'G:\dataset\BirdClef\vacation\spectrum'
    _src = r'G:\dataset\BirdClef\paper_dataset\spectrum'
    label_file = pd.read_excel(r'G:\dataset\BirdClef\vacation\label.xlsx')
    data_file = pd.read_csv(r'G:\dataset\BirdClef\paper_dataset\static.csv')

    label = list(np.asarray(label_file).reshape([-1]))
    result = data_file[data_file.Species.isin(label)]
    result.to_csv(r'G:\dataset\BirdClef\vacation\static.csv')
    '''


    # Spectrum data moving part 
    _dst = r'G:\dataset\BirdClef\vacation\spectrum'
    _src = r'G:\dataset\BirdClef\paper_dataset\spectrum'
    label_file = pd.read_excel(r'G:\dataset\BirdClef\vacation\label.xlsx')
    data_file = pd.read_csv(r'G:\dataset\BirdClef\paper_dataset\static.csv')

    label = label_file.LABEL_NAME.tolist()
    source_list = label_file.SOURCE_NAME.tolist()
    target_list = label_file.TARGET_NAME.tolist()
    for item in tqdm(data_file.iterrows()):
        item = item[1]
        if item['Species'] in label:
            if item['Species'] in source_list:
                dst = os.path.join(_dst, 'source',item['FileName'].split('.')[0])
            else:
                dst = os.path.join(_dst, 'target',item['FileName'].split('.')[0])
            src = os.path.join(_src, item['FileName'].split('.')[0])
            if not os.path.exists(dst):
                os.makedirs(dst)
            else:
                continue
            try:
                lists = os.listdir(src)
            except:
                os.rmdir(dst)
                # 对应的音频应该是被整段被认为是噪音滤除了
                print(src, ' does not exist')
                continue
            for name in lists:
                file = os.path.join(src, name)
                shutil.copy(file, dst)
    print('Spectrum Data copying Done')


    '''
    # wav file moving programming 
    dst = r'G:\dataset\BirdClef\vacation\wav'
    _src = r'G:\dataset\BirdClef\paper_dataset\wav'
    label_file = pd.read_excel(r'G:\dataset\BirdClef\vacation\label.xlsx')
    data_file = pd.read_csv(r'G:\dataset\BirdClef\paper_dataset\static.csv')

    label = list(np.asarray(label_file).reshape([-1]))
    for item in data_file.iterrows():
        item = item[1]
        if item['Species'] in label:
            src = os.path.join(_src, item['FileName'])
            shutil.copy(src, dst)
    print('Wav Data copying Done')
    '''

