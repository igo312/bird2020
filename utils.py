from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.python.saved_model import save, load
import pickle
import logging



class MODELSAVER(Callback):
    def __init__(self, filepath, monitor):
        super(MODELSAVER, self).__init__()
        self.filepath = filepath
        self.monitor = monitor

    def set_model(self, model,):
        self.model = model


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
          #logging.warning('Can save best model only with %s available, '
          #                'skipping.', self.monitor)
          raise AttributeError('Can save best model only with %s available, '
                          'skipping.', self.monitor)
        filepath = self._get_file_path(epoch, logs)
        save(self.model, filepath)
        #filepath = filepath.split('.hdf5')[0] + '.pkl'
        #pkl_file = open(filepath, 'wb')
        #pickle.dump(self.model, pkl_file)
        #pkl_file.close()


    def _get_file_path(self,epoch, logs):
        return self.filepath.format(epoch=epoch + 1, **logs)

class LossHistory(Callback):
    def __init__(self, epochs, name):
        self.epochs = epochs
        self.name = name

    def on_train_begin(self, logs={}):
        self.losses = []
        print('starttrain')

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        # 这个地方要加1
        process = float(epoch + 1) / float(self.epochs) * 100
        print(self.name + "的训练进度为" + str(process) + "loss：" + str(logs.get('loss')))

    def on_train_end(self, logs={}):
        print("endtrain")
        print(self.losses[len(self.losses) - 1])

if __name__ == '__main__':
    model = load_model(r'G:\dataset\BirdClef\vacation\Checkpoint\mobv1S1\mobv2S1-56-.hdf5')