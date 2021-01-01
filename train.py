import os
import sys
import json
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

DATA_DIR = "trainset"
BATCH_SIZE = 16
IMG_X, IMG_Y = 525, 16
GENERATIONS = 15

#randSeed = random.randint(1, 100)
randSeed = 13

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=randSeed,
    image_size=(IMG_Y, IMG_X),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=randSeed,
    image_size=(IMG_Y, IMG_X),
    batch_size=BATCH_SIZE
)


class_names = train_dataset.class_names
num_classes = len(class_names)

print(class_names)

with open(os.path.join(DATA_DIR, "classes.json"), "w") as f:
    json.dump(class_names, fp=f)

##

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

##

def _get_new_model():

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_Y, IMG_X, 3)),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),    
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        #layers.Dense(64, activation='relu'),
        layers.Dense(num_classes) # layers.Dense(5)
    ])

    optimizer='adam'
    opt_alt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=opt_alt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    return model

def _train_model(model: Sequential, name: str, generations=15):
  print("train")

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=generations
  )

  print("saved model as", name)
  model.save(name)


if __name__ == "__main__":

  #_train_model( _get_new_model(), "AIGen10.0.h5", generations=10 )
  _train_model( load_model("AIGen9.3.h5"), name="AIGen9.5.h5", generations=10 )

  sys.exit()
