import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import json
import tensorflow as tf

class PPGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PPG Signal SBP & DBP Prediction")
        self.is_fullscreen = True
        self.setup_fullscreen()

        # Load background image
        try:
            self.bg_image = Image.open(r"D:\MS_IITM\Research_work\PPG_SBP_DBP\Half_Unet\half_u_net_first_second_derivative_attention_residual\Picture1.jpg")
            self.bg_image = self.bg_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()), Image.ANTIALIAS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.bg_label = tk.Label(self.root, image=self.bg_photo)
            self.bg_label.place(relwidth=1, relheight=1)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load background image: {e}")

        self.model = None
        self.combined_values = None

        # Model path is predefined
        self.model_path = r'D:\MS_IITM\Research_work\PPG_SBP_DBP\Half_Unet\half_u_net_first_second_derivative_attention_residual\checkpoint_saved\model_checkpoint.h5'

        # Load the model automatically
        self.load_model()

        # Create UI components
        self.create_widgets()

        # Bind the Esc key to toggle fullscreen mode
        self.root.bind("<Escape>", self.toggle_fullscreen)

    def setup_fullscreen(self):
        self.root.attributes('-fullscreen', self.is_fullscreen)
        self.root.bind("<F11>", self.toggle_fullscreen)  # F11 to toggle fullscreen
        self.root.bind("<Escape>", self.toggle_fullscreen)  # Escape to exit fullscreen

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)
        if not self.is_fullscreen:
            self.root.geometry('800x600')  # Set window size when not fullscreen
        else:
            self.root.geometry('')  # Reset to fullscreen mode

    def create_widgets(self):
        # Large heading at the top center
        self.heading_label = tk.Label(self.root, text="Measuring BP from PPG Signal", font=("Helvetica", 40, "bold"), bg='lightgray', fg='black')
        self.heading_label.place(relx=0.5, rely=0.05, anchor=tk.N)

        # Load PPG Signal button
        self.load_ppg_button = tk.Button(self.root, text="Load PPG Signal", command=self.load_ppg_signal, font=("Helvetica", 14))
        self.load_ppg_button.place(relx=0.05, rely=0.3, anchor=tk.W)

        # Result button
        self.result_button = tk.Button(self.root, text="RESULT", command=self.predict_sbp_dbp, font=("Helvetica", 14))
        self.result_button.place(relx=0.05, rely=0.7, anchor=tk.CENTER)

        # Result label
        self.result_label = tk.Label(self.root, text="SBP: \nDBP: ", font=("Helvetica", 20), bg='lightgray', fg='black', relief=tk.RAISED)
        self.result_label.place(relx=0.05, rely=0.85, anchor=tk.W)

        # BP Classification
        self.bp_classification_label = tk.Label(self.root, text="BP Classification:", font=("Helvetica", 30), bg='lightgray', fg='black')
        self.bp_classification_label.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

        # NOTE section
        self.note_label = tk.Label(self.root, text="NOTE:", font=("Helvetica", 16), bg='lightgray', fg='black')
        self.note_label.place(relx=0.9, rely=0.9, anchor=tk.SE)

        self.note_text = tk.Label(self.root, text="", font=("Helvetica", 14), bg='lightgray', fg='black', wraplength=300)
        self.note_text.place(relx=0.9, rely=0.95, anchor=tk.SE)

    def load_ppg_signal(self):
        # Load PPG signal file (assumed to be in .json format as per the training script)
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.combined_values = np.array(data['combined_values'])
                self.combined_values = np.expand_dims(self.combined_values, axis=-1)  # Add channel dimension
                messagebox.showinfo("Success", "PPG Signal Loaded Successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load PPG Signal: {e}")

    def load_model(self):
        # Load the U-Net model from the predefined path
        try:
            self.model = load_model(self.model_path, custom_objects={"leaky_relu": self.leaky_relu})
            messagebox.showinfo("Success", "Model Loaded Successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def predict_sbp_dbp(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return

        if self.combined_values is None:
            messagebox.showerror("Error", "PPG Signal not loaded!")
            return

        # Predict SBP and DBP using the model
        try:
            # Assuming model expects input shape (1, 512, 3, 1)
            input_data = self.combined_values[np.newaxis, ...]
            prediction = self.model.predict(input_data)
            sbp, dbp = prediction[0]

            # Update result label
            self.result_label.config(text=f"SBP: {sbp:.2f} mmHg\nDBP: {dbp:.2f} mmHg")

            # BP Classification
            bp_classification = self.classify_bp(sbp, dbp)
            self.bp_classification_label.config(text=f"BP Classification: {bp_classification}")
            self.bp_classification_label.config(fg=self.get_classification_color(bp_classification))

            # Update NOTE section
            self.note_text.config(text=self.get_note_message(bp_classification))
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    def classify_bp(self, sbp, dbp):
        if (sbp < 90 or dbp < 60):
            return "Hypotension"
        elif ((90 <= sbp < 120) and (60 <= dbp < 80)):
            return "Normal"
        elif ((120 <= sbp < 130) and (60 <= dbp < 80)):
            return "Elevated"
        elif ((130 <= sbp < 140) or (80 <= dbp < 90)):
            return "Hypertension Stage 1"
        elif ((140 <= sbp < 180) or (90 <= dbp < 120)):
            return "Hypertension Stage 2"
        elif (sbp >= 180 or dbp >= 120):
            return "Hypertensive Crisis"
        return "Unknown"

    def get_classification_color(self, classification):
        color_map = {
            "Hypotension": "yellow",
            "Normal": "green",
            "Elevated": "orange",
            "Hypertension Stage 1": "orange",
            "Hypertension Stage 2": "red",
            "Hypertensive Crisis": "red",
            "Unknown": "black"
        }
        return color_map.get(classification, "black")

    def get_note_message(self, classification):
        message_map = {
            "Hypotension": "The BP corresponds to Hypotension, be careful...",
            "Normal": "Your BP is within the normal range.",
            "Elevated": "Your BP is elevated. Consider lifestyle changes.",
            "Hypertension Stage 1": "Hypertension Stage 1 detected. Monitor your BP regularly.",
            "Hypertension Stage 2": "Hypertension Stage 2 detected. Consult a healthcare provider.",
            "Hypertensive Crisis": "Hypertensive Crisis detected. Seek immediate medical attention.",
            "Unknown": "BP classification is unknown."
        }
        return message_map.get(classification, "No note available.")

    def leaky_relu(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

if __name__ == "__main__":
    root = tk.Tk()
    app = PPGApp(root)
    root.mainloop()
