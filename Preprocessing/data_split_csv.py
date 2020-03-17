# For getting the training data, validation data, test data.
import pandas as pd

if __name__ == '__main__':
    path = r'G:\dataset\BirdClef\vacation\source_check.csv'
    df = pd.read_csv(path)
    # Get the training data
    train = df.sample(frac=0.7)
    # Get the remaining data
    df_remain= df[~df.index.isin(train.index)]
    val = df_remain.sample(frac=2/3)
    test = df_remain[~df_remain.index.isin(val.index)]

    train.to_csv(r'G:\dataset\BirdClef\vacation\spectrum\limitspecs\train.csv')
    val.to_csv(r'G:\dataset\BirdClef\vacation\spectrum\limitspecs\validation.csv')
    test.to_csv(r'G:\dataset\BirdClef\vacation\spectrum\limitspecs\test.csv')


    # Get the source and target data from train file
    #path = r'G:\dataset\BirdClef\paper_dataset\dataset_csv\train.csv'
    #df = pd.read_csv(path)
    ## Get the training data
    #target = df.sample(frac=0.3)
    ## Get the remaining data
    #source = df[~df.index.isin(target.index)]
    #target.to_csv(r'G:\dataset\BirdClef\paper_dataset\dataset_csv\target.csv')
    #source.to_csv(r'G:\dataset\BirdClef\paper_dataset\dataset_csv\source.csv')


    '''
    # to split source and target
       path = r'G:\dataset\BirdClef\paper_dataset\static_2F.csv'
       test_path = r'G:\dataset\BirdClef\paper_dataset\目标域.xlsx'

       file = pd.read_csv(path)
       test_file = pd.read_excel(test_path)
       test_list = test_file['SpecSKey'].tolist()

       target = pd.DataFrame()

       for row in file.iterrows():
           item = row[1]
           if item['Species'] in test_list:
               temp = pd.DataFrame(item).T
               target = pd.concat([target, temp])
       source = file[~file.index.isin(target.index)]

       target.to_csv(r'G:\dataset\BirdClef\paper_dataset\dataset_transfer\target.csv')
       source.to_csv(r'G:\dataset\BirdClef\paper_dataset\dataset_transfer\source.csv')
       '''