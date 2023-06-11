# -*- coding: utf-8 -*-

import os
import scipy
import zipfile
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





# Tentukan path file zip
zip_path = 'D:/Kampus Merdeka/Bangkit 2023/capstone-project/capstone/DATASET.zip'   # Ganti dengan path file zip yang sesuai

# Ekstrak file zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('D:/Kampus Merdeka/Bangkit 2023/capstone-project/capstone')  # Ganti dengan folder tujuan ekstraksi yang diinginkan



"""**CEK FOLDER**"""
directory = 'D:/Kampus Merdeka/Bangkit 2023/capstone-project/capstone'

# Mendapatkan daftar file dan direktori dalam direktori yang ditentukan
file_list = os.listdir(directory)

# Menampilkan daftar file dan direktori
for file in file_list:
    print(file)

"""**SET DIRECTORY**"""

base_dir = 'D:/Kampus Merdeka/Bangkit 2023/capstone-project/capstone/DATASET'
train_dir = os.path.join(base_dir, 'TRAIN')
validation_dir = os.path.join(base_dir, 'VALIDATION')


#TRAIN
train_whitespot_dir = os.path.join(train_dir, 'WHITESPOT')
#train_aeromonas_dir = os.path.join(train_dir, 'AEROMONAS')
train_backterialfin_dir = os.path.join(train_dir, 'BACTERIALFIN')
train_redspot_dir = os.path.join(train_dir, 'REDSPOT')
train_healthy_dir = os.path.join(train_dir, 'HEALTHY')

#VALIDATION
validation_whitespot_dir = os.path.join(train_dir, 'WHITESPOT')
#validation_aeromonas_dir = os.path.join(train_dir, 'AEROMONAS')
validation_backterialfin_dir = os.path.join(train_dir, 'BACTERIALFIN')
validation_redspot_dir = os.path.join(train_dir, 'REDSPOT')
validation_healthy_dir = os.path.join(train_dir, 'HEALTHY')

print("Contents of base directory:")
print(os.listdir(base_dir))


"""**CEK ISI DATASET**"""
train_whitespot_fnames = os.listdir(train_whitespot_dir)
print(train_whitespot_fnames[:5])

#train_aeromonas_fnames = os.listdir(train_aeromonas_dir)
#print(train_aeromonas_fnames[:5])

train_bacterial_fnames = os.listdir(train_backterialfin_dir)
print(train_bacterial_fnames[:5])

train_healthy_fnames = os.listdir(train_healthy_dir)
print(train_bacterial_fnames[:5])

train_redspot_fnames = os.listdir(train_redspot_dir)
print(train_bacterial_fnames[:5])

#==========================================

print('total training WHITESPOT images:', len(os.listdir(train_whitespot_dir)))
#print('total training AEROMONAS images:', len(os.listdir(train_aeromonas_dir)))
print('total training BACTERIALFIN images:', len(os.listdir(train_backterialfin_dir)))
print('total training REDSPOT images:', len(os.listdir(train_redspot_dir)))
print('total training HEALTHY images:', len(os.listdir(train_healthy_dir)))





"""**DATA AUGMANTATION**"""
# Membuat objek ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Skala piksel ke range 0-1
    rotation_range=7,  # Rotasi gambar dalam range 20 derajat
    width_shift_range=0.1,  # Geser gambar secara horizontal
    height_shift_range=0.1,  # Geser gambar secara vertikal
    shear_range=0.1,  # Menggeser gambar secara shear
    zoom_range=0.1,  # Perbesar/perkecil gambar
    horizontal_flip=True,  # Flip gambar secara horizontal
    vertical_flip=True  # Flip gambar secara vertikal
)

# Menggunakan objek ImageDataGenerator untuk memuat dan augmentasi data pelatihan
train_generator = train_datagen.flow_from_directory(
    'D:/Kampus Merdeka/Bangkit 2023/capstone-project/capstone/DATASET/TRAIN',
    target_size=(224, 224),  # Ukuran gambar yang diharapkan
    # Jumlah sampel dalam setiap batch
    class_mode='categorical'  # Mode klasifikasi multikelas
)




"""**LAYER MODEL**"""

# Membangun model CNN
model = tf.keras.Sequential()

# Layer konvolusi pertama
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))




# Layer konvolusi kedua
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


# Layer konvolusi ketiga
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))




# Layer konvolusi keempat
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))


# Layer flatten
model.add(layers.Flatten())

# Layer fully connected
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))

# Layer output
model.add(layers.Dense(4, activation='softmax'))

# Menampilkan ringkasan model
model.summary()
# Menyimpan model ke file JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print("Model JSON telah disimpan.")

"""**MODEL**"""



image_width = 224
image_height = 224
batch_size= 64  # Jumlah sampel dalam setiap batch


validation_datagen = ImageDataGenerator(rescale=1./255)
# Definisikan validation data generator
val_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical')

# Definisikan optimizer kompleks
optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
   
)


#Callback
class StopTrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.9 and logs.get('val_accuracy') >= 0.9:
            print("\nTraining SESUAI NILAI ACC DAN LOSS YANG DICARI ")
            self.model.stop_training = True


# Mengompilasi model dengan optimizer kompleks
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Melatih model dan mendapatkan objek history
history = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=100, validation_data=val_generator, validation_steps=len(val_generator), callbacks=[StopTrainingCallback()])

"""**PLOT ACC DAN LOSS**"""

# Membuat plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Membuat plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""**==========================SAVE MODEL==========================**"""
def save_model_to_file(model, file_path):
    save_model(model, file_path)
    print(f"Model berhasil disimpan ke file: {file_path}")

# Simpan model ke file
save_model_to_file(model, 'D:/Kampus Merdeka/Bangkit 2023/capstone-project/capstone/path/to/model.h5')




"""**==========================PREDIKSI MODEL ==========================**"""
# Load the trained model
model = load_model('D:/Kampus Merdeka/Bangkit 2023/capstone-project/capstone/path/to/model.h5')

#CONVERT TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


# CONVETR TF JS
tfjs_target_dir = 'path/to/model.h5'
tfjs.converters.save_keras_model(model, 'path/to/tfjs_output')



# UJI PREDIKSI
def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize gamabr
    return img


def predict_image(image_path, target_size):
    img = preprocess_image(image_path, target_size)
    pred = model.predict(img)
    class_indices = train_generator.class_indices
    predicted_class = list(class_indices.keys())[np.argmax(pred)]
    confidence = np.max(pred) * 100
    return predicted_class, confidence


image_path = 'DATASET/VALIDATION/HEALTHY/budidaya-ikan-gurame.jpg'
target_size = (224, 224)  #target

predicted_class, confidence = predict_image(image_path, target_size)
print('Hasil Prediksi Kemungkinan :', predicted_class)
print('Dalam %:', confidence)