import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from U_net_512 import unet_1d

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
                    continue  # Skip files with missing keys

        return X_batch, y_batch

# Function to build the combined model (U-Net + DNN classifier)
def build_combined_model(input_size=(512, 1)):
    # Build the U-Net model
    u_net = unet_1d(input_size=input_size)

    # Output of U-Net's decoder part
    u_net_output = u_net.output

    # Flatten the output of U-Net
    flat = tf.keras.layers.Flatten()(u_net_output)

    # DNN Classifier
    dense1 = tf.keras.layers.Dense(32, activation='relu')(flat)
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
    folder = r'H:\HTIC\PulseDB\For_25k_data\3_PulseDB_25k_PPG_F_details\VitalDB_json_files_new_512\PPG_F_files'
    batch_size = 16
    input_size = (512, 1)
    epochs = 100
    final_model_path = r'H:\HTIC\PulseDB\For_25k_data\PulseDB_25k_PPG_F_details\VitalDB_json_files_new_512\PPG_F_files\Final_model\combined_model.h5'

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
        callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
                   EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=True)]
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_gen)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Save the final model
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")

    # Generate predictions
    y_true = []
    y_pred = []
    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(model.predict(X_batch), axis=1))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Hypertension Stage 1', 'Hypotension', 'Elevated', 'Hypertension Stage 2', 'Hypertensive Crisis'], yticklabels=['Normal', 'Hypertension Stage 1', 'Hypotension', 'Elevated', 'Hypertension Stage 2', 'Hypertensive Crisis'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    class_names = ['Normal', 'Hypertension Stage 1', 'Hypotension', 'Elevated', 'Hypertension Stage 2', 'Hypertensive Crisis']
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    for class_name in class_names:
        precision = report[class_name]['precision'] * 100
        recall = report[class_name]['recall'] * 100
        f1_score = report[class_name]['f1-score'] * 100
        support = report[class_name]['support']

        print(f"Class: {class_name}")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall: {recall:.2f}%")
        print(f"F1-score: {f1_score:.2f}%")
        print(f"Support: {support}")
        print()

    # Plot accuracy and validation accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
