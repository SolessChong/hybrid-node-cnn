from pylearn2.utils import serial
train_obj = serial.load_train_file('cnn.yaml')
train_obj.main_loop()