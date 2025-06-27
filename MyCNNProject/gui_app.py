import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os

# Load model and label encoder
try:
    model = load_model('best_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
except Exception as e:
    messagebox.showerror("Model Load Error", f"Error loading model or label encoder:\n{e}")
    exit()

# Predict function
def predict_image(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        # Normalize image
        std = np.std(img)
        img = (img - np.mean(img)) / std if std > 0 else img
        img = img.reshape(1, 224, 224, 1)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        class_label = le.classes_[class_index]
        confidence = prediction[0][class_index] * 100
        return f"{class_label.upper()} ({confidence:.2f}%)"
    except Exception as e:
        return f"❌ Prediction Error: {e}"

# Upload and predict
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")]
    )
    if not file_path:
        return
    try:
        # Validate image
        img = Image.open(file_path)
        img.verify()

        # Reload for display
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_panel.config(image=img_tk)
        image_panel.image = img_tk

        # Show prediction
        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}", fg="blue")
    except Exception:
        result_label.config(text="❌ Invalid image selected", fg="red")

# GUI setup
root = tk.Tk()
root.title("CT Scan Disease Classifier")
root.geometry("500x550")
root.resizable(False, False)

title_label = tk.Label(root, text="CT Scan Disease Classifier", font=("Arial", 18, "bold"))
title_label.pack(pady=15)

upload_button = tk.Button(root, text="Upload CT Image", command=upload_and_predict, font=("Arial", 12), bg="#4CAF50", fg="white")
upload_button.pack(pady=10)

image_panel = tk.Label(root)
image_panel.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()
