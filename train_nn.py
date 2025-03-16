import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

# 📌 ตั้งค่าพารามิเตอร์
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50  # เพิ่มจำนวน Epochs เพื่อให้เรียนรู้มากขึ้น
NUM_CLASSES = 67  
MAX_IMAGES_PER_CLASS = 2000  

dataset_path = r"C:\Users\ppsuc\OneDrive\Desktop\IS_project\images"
balanced_path = r"C:\Users\ppsuc\OneDrive\Desktop\IS_project\balanced_images"

# 📌 ปรับ dataset ให้สมดุล
def balance_dataset(input_dir, output_dir, max_images_per_class):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        target_class_path = os.path.join(output_dir, class_name)

        if not os.path.exists(target_class_path):
            os.makedirs(target_class_path)

        all_images = os.listdir(class_path)
        selected_images = all_images if len(all_images) <= max_images_per_class else np.random.choice(all_images, max_images_per_class, replace=False)

        for img in selected_images:
            img_src = os.path.join(class_path, img)
            img_dst = os.path.join(target_class_path, img)
            
            try:
                with Image.open(img_src) as img_file:
                    img_file = img_file.convert("RGB")  
                    img_file.save(img_dst)
            except (IOError, SyntaxError):
                print(f"❌ ลบไฟล์เสียหาย: {img_src}")
                os.remove(img_src)

    print("✅ สร้าง dataset ที่มีสมดุลมากขึ้นแล้ว!")

# 🔥 ปรับ dataset ก่อน training
balance_dataset(dataset_path, balanced_path, MAX_IMAGES_PER_CLASS)

# 📌 ใช้ Data Augmentation 
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  
)

# 📌 โหลดข้อมูลจากโฟลเดอร์ `balanced_images/`
train_data = datagen.flow_from_directory(
    balanced_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    balanced_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# 📌 คำนวณ `class_weight`
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 📌 ใช้ Pretrained Model MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# ❗️ ไม่ Train layers เดิม
base_model.trainable = False  

# 📌 สร้างโมเดลใหม่
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# 📌 ใช้ Adam Optimizer + Learning Rate Decay
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 📌 Reduce LR on Plateau (ถ้า val_loss ไม่ลด, ลด LR ลง)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
)

# 📌 Train โมเดล
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS, 
    steps_per_epoch=min(len(train_data), 100),  
    validation_steps=min(len(val_data), 50),  
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler]
)

# 📌 บันทึกโมเดล
model.save("cat_breed_cnn.h5")
print("✅ โมเดลบันทึกสำเร็จ! 🎉")

# 📌 บันทึก labels
breed_labels = list(train_data.class_indices.keys())
labels_dict = {str(i): breed_labels[i] for i in range(len(breed_labels))}

with open("labels.json", "w") as f:
    json.dump(labels_dict, f)

print("✅ บันทึก labels สำเร็จ! 🎉")
