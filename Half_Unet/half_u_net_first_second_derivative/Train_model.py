import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from Half_Unet import half_model_2D
from tensorflow.keras.layers import Dropout

# Define leaky_relu activation function
def leaky_relu(x):
    return x * tf.math.tanh(tf.math.softplus(x))

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
        X_batch = np.zeros((self.batch_size, *self.input_size))
        y_batch = np.zeros((self.batch_size, 2))  # For SBP and DBP

        for i, file_name in enumerate(batch_files):
            with open(os.path.join(self.folder, file_name), 'r') as f:
                data = json.load(f)
            combined_values = np.array(data['combined_values'])
            combined_values = np.expand_dims(combined_values, axis=-1)  # Add channel dimension
            X_batch[i] = combined_values
            y_batch[i] = [data['SBP'], data['DBP']]

        return X_batch, y_batch


# Function to build the combined model (U-Net + DNN regression)
def build_combined_model(input_size=(512, 3, 1)):  # Ensure input size includes batch dimension and channels
    # Build the U-Net model
    u_net = half_model_2D(input_size=input_size)

    # Output of U-Net's decoder part
    u_net_output = u_net.output

    # Flatten the output of U-Net
    flat = tf.keras.layers.Flatten()(u_net_output)

    # DNN Regressor
    #dense1 = tf.keras.layers.Dense(32, activation='relu')(flat)
    #dense2 = tf.keras.layers.Dense(8, activation='relu')(dense1)
    output = tf.keras.layers.Dense(2)(flat)  # predicting SBP and DBP

    # Combine U-Net and DNN Regressor into a single model
    combined_model = tf.keras.models.Model(inputs=u_net.input, outputs=output)

    # Compile the model
    combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss='mean_absolute_error',
                        metrics=['mean_absolute_error', 'mean_squared_error'])

    return combined_model


if __name__ == "__main__":
    # Define paths and parameters
    folder = r'H:\HTIC\PulseDB\For_25k_data\3_PulseDB_50K_PPG_F_details\VitalDB_json_files_PPG_F_SBP_DBP_modified\PPG_F'
    batch_size = 32
    input_size = (512, 3, 1)  # Ensure input size includes batch dimension and channels
    epochs = 250
    final_model_path = r'D:\MS_IITM\Research_work\PPG_SBP_DBP\Half_Unet\half_u_net_first_second_derivative\Final_model\combined_model.h5'
    checkpoint_dir = r'D:\MS_IITM\Research_work\PPG_SBP_DBP\Half_Unet\half_u_net_first_second_derivative\checkpoint_saved'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.h5')

    # Load file list and split into train/validation sets
    file_list = [f for f in os.listdir(folder) if f.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    # Create data generators
    train_gen = DataGenerator(folder, batch_size, input_size, train_files)
    val_gen = DataGenerator(folder, batch_size, input_size, val_files, shuffle=False)

    # Check if there is a saved model checkpoint and load it
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        # Build the combined model
        model = build_combined_model(input_size=input_size)
        print("Initialized new model.")

    print(model.summary())

    # Calculate the number of steps per epoch
    train_steps_per_epoch = len(train_files) // batch_size
    val_steps_per_epoch = len(val_files) // batch_size

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        #factor=0.5,
        #min_lr=1e-6,
        mode='auto'
    )
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

    # Train the combined model with callbacks
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[checkpoint_callback, reduce_lr_callback, early_stopping_callback]
    )

    # Evaluate the model
    val_loss, val_mae = model.evaluate(val_gen)
    print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

    # Save the final model
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")

    # Generate predictions
    y_true = []
    y_pred = []
    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        y_true.extend(y_batch)
        y_pred.extend(model.predict(X_batch))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute MAE for SBP and DBP
    sbp_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    dbp_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    print(f"SBP MAE: {sbp_mae}")
    print(f"DBP MAE: {dbp_mae}")
