# -*- encoding: utf-8 -*-

# TODO:原先的数据生成存在问题
import sys
sys.path.append(r'G:\bird2019')


from keras.model.CNN_transfer_tf import ADDA
from Preprocessing.generator.accelerate_load import tf_data_generator
import argparse

#from keras.backend import manual_variable_initialization
#manual_variable_initialization(True)

# TODO:添加batch_size， epoch的参数
# TODO：模型严重不收敛，查看模型（可能是data_gener那样的方法不OK）
# TODO：要不要调參，

#save_path = r'G:\dataset\BirdClef\paper_dataset\Checkpoint'

if __name__ == '__main__':
    # 命令行该怎么输入？:
    # python main_adda.py -参数1缩写 参数1输入 -参数2缩写 参数2输入

    ap = argparse.ArgumentParser()
    # 前面的 ‘-s’ 格式是缩写指代参数名
    ap.add_argument('-s', '--source_weights', required=False,
                    help="Path to weights file to load source model for training classification/adaptation")
    ap.add_argument('-e', '--start_epoch', type=int, default=1, required=False,
                    help="Epoch to begin training source model from")
    ap.add_argument('-n', '--discriminator_epochs', type=int, default=10000,
                    help="Max number of steps to train discriminator")
    ap.add_argument('-l', '--lr', type=float, default=0.02, help="Initial Learning Rate")
    # TODO:这里应该是在命令行中进行控制，应该选个方法是在idle中进行控制
    ap.add_argument('-f', '--train_discriminator', action='store_true',
                    help="Train discriminator model (if TRUE) vs Train source classifier")
    ap.add_argument('-a', '--source_discriminator_weights', help="Path to weights file to load source discriminator")
    ap.add_argument('-b', '--target_discriminator_weights', help="Path to weights file to load target discriminator")
    ap.add_argument('-t', '--eval_source_classifier', default=None,
                    help="Path to source classifier model to test/evaluate")
    ap.add_argument('-d', '--eval_target_classifier', default=None,
                    help="Path to target discriminator model to test/evaluate")

    ap.add_argument('-clsN','--class_num', default=10,help='Decide the class num of model will yiled')
    ap.add_argument('--mode', default='Species', help='The mode is to choose the classifying range')
    ap.add_argument('-lps', '--source_label_path', default=r'G:\dataset\BirdClef\vacation\source_check.csv')
    ap.add_argument('-lpt', '--target_label_path', default=r'G:\dataset\BirdClef\vacation\target_check.csv')
    ap.add_argument('-sp', '--source_path', default=r'G:\dataset\BirdClef\vacation\train_file\source')
    ap.add_argument('-tp', '--target_path', default=r'G:\dataset\BirdClef\vacation\train_file\target')
    ap.add_argument('-SSP', '--source_spec_path', default=r'G:\dataset\BirdClef\vacation\spectrum\source')
    ap.add_argument('-TSP', '--target_spec_path', default=r'G:\dataset\BirdClef\vacation\spectrum\target')
    ap.add_argument('--batch_size', default=24, type=int, help='The input size that every iteration go through in.')
    args = ap.parse_args()

    # Get the data generator
    # class_num should be a num or None
    source_gener = tf_data_generator(label_path=args.source_label_path,
                                     csv_dir=args.source_path,
                                     spec_dir=args.source_spec_path,
                                     class_num=args.class_num)
    # limit_species=['obsoletum', 'sulphurescens']
    train_source, train_source_num = source_gener.gen_return('train', args.batch_size, True)
    val_source, val_source_num = source_gener.gen_return('validation', args.batch_size, False)
    test_source, test_source_num = source_gener.gen_return('test', args.batch_size, False)

    # Training procedure
    adda = ADDA(args.lr, args.mode, class_num=args.class_num, Img_size=[256, 512])
    adda.define_source_encoder()
    #a = time.time()
    #  a,b = next(source_gen)
    #b = time.time()
    #print('The time consumed is {}'.format(b-a))
    #next(val_gen)
    #print('The train_discriminator is {}'.format(args.train_discriminator))
    # Train discriminator model (if TRUE) vs Train source classifier
    # 1.Train the Classifier
    # args.source_weights = r'G:\dataset\BirdClef\vacation\Checkpoint\bird202040.hdf5'
    if not args.train_discriminator:
        if args.eval_source_classifier is None:
            if args.source_weights is None:
                print('training a new model.')
            print("training dataset's size is {}".format(train_source_num))
            model = adda.get_source_classifier(adda.source_encoder, args.source_weights)
            # we can enlarget the steps_per_epoch to see whether the model can converge
            adda.train_source_model(train_source, None, model,name='bird2020num10',
                                    epochs=1024, steps_per_epoch=1.5*train_source_num//args.batch_size,\
                                    validation_steps=None,\
                                    start_epoch=args.start_epoch - 1,
                                    save_interval=30,
                                    C_state=True,
                                    lr=0.005
                                   )
            print('trainging Done, processing evaluating')
            adda.eval_source_classifier(test_source, model, 'source')
        else:
            model = adda.get_source_classifier(adda.source_encoder, args.eval_source_classifier)
            adda.eval_source_classifier(test_source, model, 'source')
            #adda.eval_source_classifier(model, 'target')
    # adda.define_target_encoder(args.source_weights)
    # 2.Train the Discriminator
    else :
        target_gener = tf_data_generator(label_path=args.target_label_path,
                                         csv_dir=args.target_path,
                                         spec_dir=args.target_spec_path,
                                         class_num=2)
        train_target, train_target_num = source_gener.gen_return('train', args.batch_size, True)
        val_target, val_target_num = source_gener.gen_return('validation', args.batch_size, False)
        test_target, test_target_num = source_gener.gen_return('test', args.batch_size, False)

        args.source_weights = r'G:\dataset\BirdClef\vacation\Checkpoint\bird202040.hdf5'
        adda.define_target_encoder(args.source_weights)
        # batch_data = (next(train_source), next(train_target))
        adda.train_target_discriminator(train_source, train_target,
                                        epochs=args.discriminator_epochs,
                                        num_batches=train_target_num//args.batch_size,
                                        source_model=args.source_weights,
                                        src_discriminator=args.source_discriminator_weights,
                                        tgt_discriminator=args.target_discriminator_weights,
                                        start_epoch=args.start_epoch - 1)
    # print()
    #if args.eval_target_classifier is not None:
    #   adda.eval_target_classifier(args.eval_source_classifier, args.eval_target_classifier)

    # TODO:没进行eval

