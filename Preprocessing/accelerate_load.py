# To make a tensorflow dataset generator to make training faster

import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

from Preprocessing.generator_for_tf import Data_Gener

if __name__ == '__main__':
    # if the data can fits our memory, we can use tf.data.Dataset.from_tensor_slices
    # To make a python generator
    label_path = r'G:\dataset\BirdClef\vacation\source.csv'
    train_path = r'G:\dataset\BirdClef\vacation\train_file\source\source_train.csv'
    val_path = r'G:\dataset\BirdClef\vacation\train_file\source\source_val.csv'
    test_path = r'G:\dataset\BirdClef\vacation\train_file\source\source_val.csv'
    spec_path = r'G:\dataset\BirdClef\paper_dataset\spectrum'
    batchsize = 16
    class_num = 10
    source_gener = Data_Gener(mode='Species', Img_size=[256, 512],
                              label_path=label_path,
                              limit_species=class_num)
    #[train_source, val_source, test_source], lens = source_gener.data_gener(data_file_path=data_file_path1,
    #                                                                       BatchSize=batchsize, spec_path=spec_path,
    #                                                                      aug=True)

    # The Dataset.from_generator constructor converts the python generator to a fully functional tf.data.Dataset.
    # TODO:how to return a batch to see what happend?
    # TODO:how to do multiprocessing job? cpu_count = 4
    '''
    tf_gener = tf_gener.repeat() # still need .repaet ?
    tf_gener.batch() # return a batch?
    '''
    tf_gener = tf.data.Dataset.from_generator(source_gener.data_gener, output_types=(tf.float64,tf.uint8),
                                              output_shapes=((256,512), (class_num)),
                                              args=(train_path, spec_path, True),
                                              )
    # gen = tf_gener.repeat()
    data = []
    gen = tf_gener.repeat().batch(16)
    iterator = gen.make_initializable_iterator()
    x, y = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        a = sess.run(x)
        for x in tf_gener.repeat().take(10):
            data.append(x)

