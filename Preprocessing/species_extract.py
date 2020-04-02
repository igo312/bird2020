# 以索引形式提取数据， 返回一个含有完整信息的limit_species文件以及一个label文件
import pandas as pd

if __name__ == '__main__':
    spcies_num = 30
    mode = 'source'

    static_path = r'G:\dataset\BirdClef\paper_dataset\static.csv'
    un_path = r'G:\dataset\BirdClef\paper_dataset\unbanlaced_static.xlsx'
    save_path = r'G:\dataset\BirdClef\vacation\ls'

    static_df = pd.read_csv(static_path)
    un_df = pd.read_excel(un_path)
    limit_df =[]

    # S:means species, G:means genus
    # wavSkey = un_df['WavSKey']
    # wavGkey = un_df['WavGKey']
    wavSkey = un_df['WavSKey']
    wavkey = wavSkey[-30:].to_list()

    for item in static_df.iterrows():
        item = item[1]
        if item['Species'] in wavkey:
            limit_df.append(item)

    wavkey = pd.DataFrame(wavkey)
    wavkey.to_csv(save_path + mode + '_label' + str(spcies_num) + '.csv')

    limit_df = pd.DataFrame(limit_df)
    limit_df.to_csv(save_path+ mode + '_limit'+str(spcies_num) + '.csv', index=False)

    print('Done')