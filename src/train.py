import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

#provide path to dataset
DATASET_PATH = 'data/processed/cats-breads'

#validate if path exists
assert os.path.exists(DATASET_PATH), "dataset path not found!"

#Training parameters
IMG_SIZE =  (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

#Data pipline and augmentation
train_datagen = ImageDataGenerator (
    rescale=1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = 0.2 #20% validation
)

#Training data generator
train_generator = train_datagen.flow_from_directory (
    DATASET_PATH,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'training',
    seed = 42
)

#Validation data generator
val_generator = train_datagen.flow_from_directory (
    DATASET_PATH,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'validation',
    seed = 42
)

#number of breeds
NUM_CLASSES = len(train_generator.class_indices)

#Model
model = models.Sequential([
    #Extracting features
    Input(shape = (*IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    #Classification
    layers.Flatten(),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(NUM_CLASSES, activation = 'softmax')
])

#Compile the model
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#early stop to avoid overfitting from too many epochs
early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    restore_best_weights = True
)

#Training the model
print("Training model!")
history = model.fit(
    train_generator,
    epochs = EPOCHS,
    validation_data = val_generator,
    callbacks = [early_stop],
    verbose = 1 #stylistic, shows progress bar
)

os.makedirs('models', exist_ok = True)
model.save('models/cat_breed_classifier.h5')
print("Training complete!!!")