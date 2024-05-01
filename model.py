import os
import numpy as np
import pandas as pd  # Přidání importu pandas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

dataset_home = 'data/'

train_files = [os.path.join(dataset_home, 'train/', f) for f in os.listdir(os.path.join(dataset_home, 'train/')) if f.endswith('.jpg')]

train_labels = [1 if 'dog' in os.path.basename(f) else 0 for f in train_files]

print('Trenovaci data: ', len(train_files))

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

datagen = ImageDataGenerator(rescale=1.0/255.0)

train_df = pd.DataFrame({'filename': train_files, 'class': train_labels})
train_df['class'] = train_df['class'].astype(str)

train_it = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=64,
    class_mode='binary'
)

model = define_model()

history = model.fit(train_it, steps_per_epoch=len(train_it), epochs=20, verbose=0)

model.save('model.h5')
