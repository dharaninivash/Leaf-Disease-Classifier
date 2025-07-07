import os
import shutil
import random
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import threading
import cv2  # For marking diseased areas

# Create required folders
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)
os.makedirs("model", exist_ok=True)

labels = ['Diseased', 'Healthy']
model_path = "model/cnn_model.h5"

# GUI Setup
root = tk.Tk()
root.title("🌿 Leaf Disease Classifier")
root.geometry("500x500")
root.resizable(False, False)

healthy_path = tk.StringVar()
diseased_path = tk.StringVar()

# To hold images globally
display_img_tk = None
marked_img_tk = None

# ===== Helper Functions =====

def split_data(source, train_dir, test_dir, label, split_ratio=0.8):
    files = os.listdir(source)
    random.shuffle(files)
    split = int(len(files) * split_ratio)
    train_files = files[:split]
    test_files = files[split:]

    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(source, f), os.path.join(train_dir, label, f))
    for f in test_files:
        shutil.copy(os.path.join(source, f), os.path.join(test_dir, label, f))

def train_model():
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'

    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)

    split_data(healthy_path.get(), train_dir, test_dir, "Healthy")
    split_data(diseased_path.get(), train_dir, test_dir, "Diseased")

    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_dir, target_size=(128, 128), class_mode='categorical', batch_size=32)

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir, target_size=(128, 128), class_mode='categorical', batch_size=32)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    progress['value'] = 20
    root.update()

    model.fit(train_gen, validation_data=test_gen, epochs=5)

    model.save(model_path)
    progress['value'] = 100
    messagebox.showinfo("✅ Done", "Model trained and saved successfully!")
    predict_button['state'] = 'normal'

def select_healthy():
    path = filedialog.askdirectory(title="Select Healthy Images Folder")
    if path:
        healthy_path.set(path)
        healthy_label.config(text=os.path.basename(path))

def select_diseased():
    path = filedialog.askdirectory(title="Select Diseased Images Folder")
    if path:
        diseased_path.set(path)
        diseased_label.config(text=os.path.basename(path))

def threaded_training():
    if not healthy_path.get() or not diseased_path.get():
        messagebox.showerror("❌ Error", "Please select both folders.")
        return
    progress['value'] = 0
    threading.Thread(target=train_model).start()

def predict_leaf():
    global display_img_tk, marked_img_tk

    if not os.path.exists(model_path):
        messagebox.showerror("Model Missing", "Please train the model first.")
        return

    img_path = filedialog.askopenfilename(
        title="Select a Leaf Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not img_path:
        return

    model = load_model(model_path)
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    idx = np.argmax(pred)
    result_text = f"Prediction: {labels[idx]} ({pred[0][idx] * 100:.2f}%)"

    # Show Original Image
    display_img = Image.open(img_path).resize((200, 200))
    display_img_tk = ImageTk.PhotoImage(display_img)

    img_window = tk.Toplevel(root)
    img_window.title("Prediction Result")
    img_window.geometry("500x550")

    tk.Label(img_window, text=result_text, font=("Arial", 14)).pack(pady=10)
    tk.Label(img_window, image=display_img_tk).pack(pady=10)

    # Simulated Marking (Edge Detection)
    original = cv2.imread(img_path)
    original = cv2.resize(original, (300, 300))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    marked = cv2.addWeighted(original, 0.8, edge_colored, 0.5, 0)
    cv2.imwrite("temp_marked.jpg", marked)

    marked_img = Image.open("temp_marked.jpg").resize((300, 300))
    marked_img_tk = ImageTk.PhotoImage(marked_img)

    tk.Label(img_window, text="🟠 Marked Suspected Regions (Simulated)", font=("Arial", 12)).pack()
    tk.Label(img_window, image=marked_img_tk).pack()

    img_window.mainloop()

# ===== GUI Layout =====

tk.Label(root, text="🌿 Leaf Disease Classifier", font=("Arial", 18)).pack(pady=15)

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="Select Healthy Folder", command=select_healthy, bg="#4CAF50", fg="white", width=20).grid(row=0, column=0, padx=10, pady=5)
healthy_label = tk.Label(frame, text="No folder selected", width=30)
healthy_label.grid(row=0, column=1)

tk.Button(frame, text="Select Diseased Folder", command=select_diseased, bg="#F44336", fg="white", width=20).grid(row=1, column=0, padx=10, pady=5)
diseased_label = tk.Label(frame, text="No folder selected", width=30)
diseased_label.grid(row=1, column=1)

train_btn = tk.Button(root, text="🚀 Train CNN Model", command=threaded_training, font=("Arial", 12), bg="#2196F3", fg="white", width=30)
train_btn.pack(pady=20)

progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress.pack(pady=10)

predict_button = tk.Button(root, text="📸 Predict Leaf Image", command=predict_leaf, font=("Arial", 12), bg="orange", fg="white", width=30)
predict_button.pack(pady=20)
predict_button['state'] = 'disabled'

tk.Label(root, text="By Dharani Nivash • Offline Desktop App", font=("Arial", 9)).pack(side=tk.BOTTOM, pady=15)

root.mainloop()
