import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

dataset = 'data/'

with open('data/list.txt', 'r') as file:
    zvirata = file.read().splitlines()

trainf = []
trainl = []
for zvire in zvirata:
    obrazky = [os.path.join(dataset, 'train', zvire, f) for f in os.listdir(os.path.join(dataset, 'train', zvire)) if f.endswith('.jpg')]
    trainf.extend(obrazky)
    trainl.extend([zvire] * len(obrazky))

print('Trenovaci data: ', len(trainf))
print('Tridy: ', zvirata)

def dfn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(len(zvirata), activation='softmax'))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

datagen = ImageDataGenerator(rescale=1.0/255.0)
train_df = pd.DataFrame({'filename': trainf, 'class': trainl})

train_it = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=64,
    class_mode='categorical'
)

model = dfn()
history = model.fit(train_it, steps_per_epoch=len(train_it), epochs=20, verbose=1)
model.save('model.h5')
