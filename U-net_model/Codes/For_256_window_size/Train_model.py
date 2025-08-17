# Train_model.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from U_net import unet_1d

# Define the data generator class
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder, batch_size, input_size, file_list, shuffle=True):
        self.folder = folder
        self.batch_size = batch_size
        self.input_size = input_size
        self.file_list = file_list
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.file_list[i] for i in batch_indexes]
        return self.__data_generation(batch_files)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_files):
        X_batch = np.empty((self.batch_size, *self.input_size))
        y_batch = np.empty((self.batch_size, 6))  # assuming 6 output classes

        for i, filename in enumerate(batch_files):
            file_path = os.path.join(self.folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                try:
                    X_batch[i, :, 0] = data['PPG_values']
                    bp_class = data['BP_Classification']
                    bp_class_index = {
                        'Normal': 0,
                        'Hypertension Stage 1': 1,
                        'Hypotension': 2,
                        'Elevated': 3,
                        'Hypertension Stage 2': 4,
                        'Hypertensive Crisis': 5
                    }
                    y_batch[i] = tf.keras.utils.to_categorical(bp_class_index[bp_class], num_classes=6)
                except KeyError as e:
                    print(f"Missing key {e} in file: {filename}")
                    # Handle missing key gracefully, e.g., by skipping the file or using a default class
                    continue  # Skip files with missing keys

        return X_batch, y_batch
# Function to build the combined model (U-Net + DNN classifier)
def build_combined_model(input_size=(256, 1)):
    # Build the U-Net model
    u_net = unet_1d(input_size=input_size)

    # Freeze U-Net layers
    #for layer in u_net.layers:
     #   layer.trainable = False

    # Output of U-Net's decoder part
    u_net_output = u_net.get_layer('conv1d_22').output  # Replace 'conv9' with the actual name of last decoder layer

    # Flatten the output of U-Net
    flat = tf.keras.layers.Flatten()(u_net_output)

    # DNN Classifier
    dense1 = tf.keras.layers.Dense(64, activation='relu')(flat)
    output = tf.keras.layers.Dense(6, activation='softmax')(dense1)  # 6 classes for BP classification

    # Combine U-Net and DNN Classifier into a single model
    combined_model = tf.keras.models.Model(inputs=u_net.input, outputs=output)

    # Compile the model
    combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    return combined_model

# Main script for training
if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Define paths and parameters
    folder = r'H:\HTIC\PulseDB\VitalDB_PulseDB\VitalDB_json_files_new\json_modified'
    batch_size = 32
    input_size = (256, 1)
    epochs = 200
    final_model_path = r'H:\HTIC\PulseDB\VitalDB_PulseDB\VitalDB_json_files_new\combined_model.h5'

    # Load file list and split into train/validation sets
    file_list = [f for f in os.listdir(folder) if f.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    # Create data generators
    train_gen = DataGenerator(folder, batch_size, input_size, train_files)
    val_gen = DataGenerator(folder, batch_size, input_size, val_files, shuffle=False)

    # Build the combined model
    model = build_combined_model(input_size=input_size)
    print(model.summary())

    # Calculate the number of steps per epoch
    train_steps_per_epoch = len(train_files) // batch_size
    val_steps_per_epoch = len(val_files) // batch_size

    # Train the combined model with callbacks
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1),
                   EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)]
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_gen)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Save the final model
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")
