from keras.utils import to_categorical
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np

EPOCHS = 100

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def reshape_data(data: np.ndarray, channel: str):
    n_img, img_row, img_column = data.shape

    if channel == 'channels_first':
        temp_data = data.reshape(n_img, 1, img_row, img_column)
        input_shape = (1, img_row, img_column)

    elif channel == 'channels_last':
        temp_data = data.reshape(n_img, img_row, img_column, 1)
        input_shape = (img_row, img_column, 1)
    else:
        raise ValueError('Invalid channel. \
                         Allowed values are "channels_first" \
                         and "channels_last".')

    return (temp_data.astype('float32'),
            input_shape)


x_train, input_shape_train = reshape_data(x_train, 'channels_last')
x_test, input_shape_test = reshape_data(x_test, 'channels_last')

x_train /= 255
x_test /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inputs = Input(shape=input_shape_train)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = Dropout(0.1)(x)
x = MaxPooling2D(pool_size=[2, 2])(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

modelo = Model(inputs, outputs)
modelo.summary()

modelo.compile(loss=['categorical_crossentropy'],
               optimizer='adam')

checkpoint = ModelCheckpoint(filepath='checkpoint',
                             monitor='val_loss',
                             mode='auto',
                             save_best_only=True,
                             format='h5')

modelo.fit(x=x_train,
           y=y_train,
           epochs=EPOCHS,
           batch_size=64,
           callbacks=[checkpoint],
           validation_data=(x_train, y_train),
           validation_split=0.1)

modelo.save(filepath='modelo/modelo1.h5',
            save_format='h5',
            overwrite=True)

with open('jsons/modelo.json', 'w') as arquivo_json:
    arquivo_json.write(modelo.to_json())
