import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os
import streamlit as st
from train_ml import train_and_save_metrics


train_and_save_metrics(noise_std=2.0)

import json



# โหลดโมเดล Machine Learning สำหรับทำนายบ้านใน Harry Potter
RF_MODEL_PATH = "house_rf.pkl"
LR_MODEL_PATH = "house_lr.pkl"
rf_model = joblib.load(RF_MODEL_PATH)
lr_model = joblib.load(LR_MODEL_PATH)

# โหลดโมเดล Neural Network สำหรับทำนายสายพันธุ์แมว
NN_MODEL_PATH = "cat_breed_cnn.h5"
nn_model = load_model(NN_MODEL_PATH)

# โหลด labels จากไฟล์ JSON ที่บันทึกไว้
LABELS_PATH = "labels.json"
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        breed_labels = json.load(f)
else:
    breed_labels = {str(i): f"Breed {i}" for i in range(67)}  # ใช้ labels ชั่วคราวแทน

# ฟังก์ชันทำนายบ้านใน Hogwarts
def predict_house(features):
    rf_prediction = rf_model.predict([features])[0]
    lr_prediction = lr_model.predict([features])[0]
    return rf_prediction, lr_prediction

# ฟังก์ชันทำนายสายพันธุ์แมวจากรูปภาพ
def predict_cat_breed(image_file):
    IMG_SIZE = 128
    img = image.load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = nn_model.predict(img_array)
    breed_index = str(np.argmax(prediction))
    return breed_labels.get(breed_index, "Unknown")

rating_bravery = {
    "0": "I'd rather be a portrait on the wall than face what's beyond the door.",
    "1": "I stepped onto the Hogwarts Express, but I'm still debating whether this was a good idea.",
    "2": "I'm willing to follow my friends into the Forbidden Forest, but only if I'm at the very back.",
    "3": "I'll face danger if I must, but only after exhausting every escape plan.",
    "4": "I break the rules, but only when no professors or ghosts are watching.",
    "5": "My hands may shake, but I'll stand my ground when it matters.",
    "6": "I won't start the trouble, but if someone dares me, I'm already halfway through the challenge.",
    "7": "Danger is not an obstacle, but a challenge.",
    "8": "My first instinct isn't to run. It's to protect others.",
    "9": "My bravery echoes that of the greatest wizards before me, and people whisper my name with admiration.",
    "10": "Fear exists, but it never controls me. Even in the face of death, I stand tall."
}


rating_intelligence = {
    "0": "Thinking? What's that? Even a simple spell confuses me.",
    "1": "I show up to class, but I'm too busy thinking about what's for dinner.",
    "2": "I show up to class, but my mind is in Hogsmeade instead of on the lesson.",
    "3": "I survive school by copying my friend's notes five minutes before class.",
    "4": "I study when necessary, but if there's a Quidditch match, books can wait.",
    "5": "I may not be top of the class, but I can outthink a Cornish Pixie.",
    "6": "I spend extra hours in the library, but I still panic during exams.",
    "7": "I don't just cast spells, I want to know why they work.",
    "8": "I can solve problems even before they happen, and no challenge ever truly surprises me.",
    "9": "I understand magic on a level few others do. Even the Restricted Section has little to teach me.",
    "10": "I know things that even the most experienced wizards dare not speak of."
}


rating_loyalty = {
    "0": "I don't seek power, titles, or attention—I'm happiest serving others.",
    "1": "I have potential, but I prefer to stay in the background, unnoticed and unbothered.",
    "2": "I may have leadership qualities, but I don't recognize them in myself—yet.",
    "3": "I lead with kindness, though I rarely take the spotlight.",
    "4": "I prefer to stay behind the throne, offering wisdom while others wear the crown.",
    "5": "I step up when needed, defending those who cannot defend themselves.",
    "6": "I'm learning to command respect, though I still seek approval from others.",
    "7": "I have a vision of greatness, and I won't stop until the world sees it too.",
    "8": "I am loyal to the pursuit of truth, for knowledge commands respect and influence.",
    "9": "I've fought for my place at the top, and my people trust me with their lives.",
    "10": "I want my name to be remembered in history."
}


rating_ambition = {
    "0": "I have no grand dreams—I'm content with where I am, and I don't seek more.",
    "1": "I think about success, but taking action seems like too much effort.",
    "2": "I have small goals, but I'm easily discouraged when obstacles arise.",
    "3": "I work toward my goals, but only if they don't push me too far outside my comfort zone.",
    "4": "I aim high, but I hesitate to take risks that might lead to failure.",
    "5": "I want to achieve great things, and I take steps to make it happen.",
    "6": "I am willing to put in the effort, but I still care about how others perceive me.",
    "7": "I see challenges as stepping stones to success, and I won't stop until I get what I want.",
    "8": "I set my sights on greatness, and I will do whatever it takes to reach it.",
    "9": "I don't wait for opportunities—I create them, and I refuse to let anything stand in my way.",
    "10": "Power, legacy, and success are not just my goals—they are my destiny."
}


rating_dark_arts_knowledge = {
    "0": "I avoid anything related to the Dark Arts—I won't even read about it.",
    "1": "I know the Dark Arts exist, but I have no interest in learning more.",
    "2": "I've heard stories about Dark Magic, but I would never dare to explore it myself.",
    "3": "I understand the basics of the Dark Arts, but I only see them as dangerous and forbidden.",
    "4": "I've read about Dark Magic out of curiosity, but I would never use it.",
    "5": "I acknowledge that the Dark Arts hold power, but I believe they should be studied responsibly.",
    "6": "I study the Dark Arts not to use them, but to understand how to defend against them.",
    "7": "I know spells that many would consider questionable, but I use them with caution.",
    "8": "I believe the Dark Arts are just another form of magic—power is power.",
    "9": "I have mastered the Dark Arts to the point where even seasoned wizards would fear my knowledge.",
    "10": "There is no curse, hex, or dark spell beyond my understanding—I wield darkness like a true master."
}


rating_quidditch_skills = {
    "0": "I can barely sit on a broomstick without falling off.",
    "1": "I know how to fly, but my balance is terrible.",
    "2": "I can stay in the air, but catching or dodging anything is a problem.",
    "3": "I can manage basic flying, but I wouldn't trust myself in an actual match.",
    "4": "I can play for fun, but I wouldn't make it onto the house team.",
    "5": "I have decent flying skills, and I can hold my own in a casual game.",
    "6": "I can play in competitive matches, but I still have room for improvement.",
    "7": "I'm skilled enough to be a valuable player on my house team.",
    "8": "I play like a professional, and I could make it to the big leagues if I tried.",
    "9": "I am one of the best players Hogwarts has ever seen—scouts keep an eye on me.",
    "10": "I play at a level that rivals the greatest Quidditch legends of all time."
}


rating_dueling_skills = {
    "0": "I wouldn't even know which end of my wand to point in a duel.",
    "1": "I know a few spells, but I'd probably drop my wand under pressure.",
    "2": "I can cast basic defensive spells, but I panic in a real fight.",
    "3": "I have some dueling knowledge, but my reflexes need work.",
    "4": "I can hold my own in a friendly duel, but I lack advanced techniques.",
    "5": "I know enough spells to defend myself, but I wouldn't challenge a champion.",
    "6": "I have solid dueling skills and can outmatch most casual duelists.",
    "7": "I am a skilled duelist and can win most one-on-one battles.",
    "8": "My reflexes and spellwork are sharp—I could duel professionally.",
    "9": "I am a master duelist, feared and respected in any magical duel.",
    "10": "Legends speak of my dueling prowess—only the greatest dare to challenge me."
}


rating_creativity = {
    "0": "I struggle to think outside the box—following instructions is my comfort zone.",
    "1": "I occasionally have new ideas, but I mostly stick to what I know.",
    "2": "I can come up with simple solutions, but I rarely think beyond the obvious.",
    "3": "I like experimenting, but I often doubt my own ideas.",
    "4": "I enjoy creative thinking, but I sometimes need inspiration to get started.",
    "5": "I can think of unique solutions, but I still refine ideas rather than create new ones.",
    "6": "I often come up with innovative ideas and love finding unconventional approaches.",
    "7": "Creativity is second nature to me—I can solve problems in ways others wouldn't consider.",
    "8": "I thrive on originality and can turn ordinary ideas into something extraordinary.",
    "9": "My mind is a wellspring of creativity—I see possibilities where others see limitations.",
    "10": "I am a true visionary, capable of redefining magic, art, and innovation itself."
}



# สร้างแถบเลือกหน้า
page = st.sidebar.selectbox("# Page selection", ["Machine Learning", "🧙🏻‍♂️ The Sorting Hat", "Neural Network", "🐈 Guess the cat"])

# หน้าแรก
if page == "Machine Learning":
    st.title("Predicting Hogwarts House Using Machine Learning 🏰")

    st.header("Purpose")
    st.write(
        "This project aims to build a machine learning model that predicts "
        "the Hogwarts house of a student based on personality traits and skills. "
        "By analyzing these attributes, I gain insights into how certain qualities influence house assignments."
    )

    st.header("Problem Type")
    st.write(
        "I approach this as a **classification problem**, where each student "
        "is assigned to one of four possible houses. The model learns patterns from the labeled data to make accurate predictions."
    )

    st.header("Source Data")
    st.write(
        "The dataset used for this model is sourced from Kaggle. "
        "You can find the dataset here: [Harry Potter Sorting Dataset](https://www.kaggle.com/datasets/sahityapalacharla/harry-potter-sorting-dataset)"
    )
    st.markdown(
        """
        The dataset includes various attributes related to personality and skills:
        - **Bravery**: Measure of courage and boldness.
        - **Intelligence**: Level of knowledge and problem-solving ability.
        - **Loyalty**: Dedication to friends and ideals.
        - **Ambition**: Drive to achieve goals and succeed.
        - **Dark Arts Knowledge**: Understanding of forbidden magic.
        - **Quidditch Skills**: Flying and gameplay expertise.
        - **Dueling Skills**: Magical combat effectiveness.
        - **Creativity**: Ability to think outside the box.
        """
    )

    st.header("Data Preparation")
    st.write("Steps I took to process the data:")
    st.markdown(
        """
        1. **Filtering**: I only kept complete responses to maintain data integrity.
        2. **Feature Selection**: I focused on key attributes that contribute to classification.
        3. **Data Scaling**: Standardization was applied to improve model performance.
        4. **Noise Injection**: I added Gaussian noise to each numerical feature to test the model's robustness. Values were clipped to remain within realistic bounds (1–10).
        5. **Train-Test Split**: I split the dataset into 80% training and 20% testing data.
        """
    )

    st.header("Model")
    st.write("I used two machine learning models to predict Hogwarts houses:")
    st.markdown(
        """
        - **Random Forest Classifier**: An ensemble learning model combining multiple decision trees.
        - **Logistic Regression**: A simpler model that predicts the probability of each class.
        """
    )

    st.header("Model Performance")
    st.write("I evaluated the model using the following metrics:")
    st.markdown(
        """
        - **Accuracy**: The percentage of correct predictions.
        - **F1 Score**: The harmonic mean of precision and recall.
        - **Confusion Matrix**: Visualizes misclassification patterns.
        """
    )

    st.header("Deployment")
    st.write(
        "After training and evaluating, I deployed the models using Streamlit. "
        "Users can interactively input their traits to receive a predicted Hogwarts house."
    )

    st.header("Machine Learning Code")
    st.code('''import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train_and_save_metrics(noise_std=0.5):
    # Load the dataset
    dataset_file = "house.csv"
    df = pd.read_csv(dataset_file)

    # Specify the numeric features
    features = [
        "Bravery", 
        "Intelligence", 
        "Loyalty", 
        "Ambition",
        "Dark Arts Knowledge", 
        "Quidditch Skills", 
        "Dueling Skills", 
        "Creativity"
    ]
    
    # Create a copy of the data for noise injection
    df_noisy = df.copy()

    # Inject Gaussian noise into each feature and clip values between 1 and 10
    for col in features:
        noise = np.random.normal(0, noise_std, size=df_noisy.shape[0])
        df_noisy[col] = (df_noisy[col] + noise).clip(lower=1, upper=10)

    X = df_noisy[features]
    y = df_noisy["House"]

    # Shuffle the dataset
    df_noisy = df_noisy.sample(frac=1, random_state=None).reset_index(drop=True)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
    joblib.dump(rf_model, "house_rf.pkl")

    # Train Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')
    joblib.dump(lr_model, "house_lr.pkl")

    # Save performance metrics in session state
    if "rf_accuracy" not in st.session_state:
        st.session_state["rf_accuracy"] = rf_accuracy
        st.session_state["rf_f1"] = rf_f1
        st.session_state["lr_accuracy"] = lr_accuracy
        st.session_state["lr_f1"] = lr_f1

if __name__ == "__main__":
    train_and_save_metrics(noise_std=2.0)''', language='python')


    
# หน้าที่ 2: ทำนายบ้านใน Hogwarts (ML)
elif page == "🧙🏻‍♂️ The Sorting Hat":
    st.title("Select the house that suits you by Machine Learning")
    st.write("### Be intentional about answering these questions.")
    
    bravery = st.selectbox("What would you do if you were in a dangerous situation?", list(rating_bravery.values()))
    intelligence = st.selectbox("What type of person are you in class?", list(rating_intelligence.values()))
    loyalty = st.selectbox("Do you prefer to stand in the spotlight, lead from behind, or dedicate yourself to others?", list(rating_loyalty.values()))
    ambition = st.selectbox("Do you have goals in life?", list(rating_ambition.values()))
    dark_arts_knowledge = st.selectbox("How much do you know about the dark arts?", list(rating_dark_arts_knowledge.values()))
    quidditch_skills = st.selectbox("How good are you at quidditch?", list(rating_quidditch_skills.values()))
    dueling_skills = st.selectbox("How confident are you in your dueling abilities?", list(rating_dueling_skills.values()))
    creativity = st.selectbox("How often do you come up with unique or unconventional solutions to problems?", list(rating_creativity.values()))
    
    if st.button("Done"):
        user_data = [
            int(list(rating_bravery.keys())[list(rating_bravery.values()).index(bravery)]),
            int(list(rating_intelligence.keys())[list(rating_intelligence.values()).index(intelligence)]),
            int(list(rating_loyalty.keys())[list(rating_loyalty.values()).index(loyalty)]),
            int(list(rating_ambition.keys())[list(rating_ambition.values()).index(ambition)]),
            int(list(rating_dark_arts_knowledge.keys())[list(rating_dark_arts_knowledge.values()).index(dark_arts_knowledge)]),
            int(list(rating_quidditch_skills.keys())[list(rating_quidditch_skills.values()).index(quidditch_skills)]),
            int(list(rating_dueling_skills.keys())[list(rating_dueling_skills.values()).index(dueling_skills)]),
            int(list(rating_creativity.keys())[list(rating_creativity.values()).index(creativity)])
        ]
        
        rf_prediction, lr_prediction = predict_house(user_data)
        
        # ดึงค่าประสิทธิภาพของโมเดลจาก st.session_state
        rf_accuracy = st.session_state.get("rf_accuracy", None)
        rf_f1 = st.session_state.get("rf_f1", None)
        lr_accuracy = st.session_state.get("lr_accuracy", None)
        lr_f1 = st.session_state.get("lr_f1", None)
        
        st.subheader("Results of selection")
        st.write(f"##### Random Forest Prediction: *{rf_prediction}*")
        if rf_accuracy is not None and rf_f1 is not None:
            st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}  ,  F1-Score: {rf_f1:.2f}")
        else:
            st.write("Random Forest performance metrics not available.")
            
        st.write(f"##### Logistic Regression Prediction: *{lr_prediction}*")
        if lr_accuracy is not None and lr_f1 is not None:
            st.write(f"Logistic Regression Accuracy: {lr_accuracy:.2f}  ,  F1-Score: {lr_f1:.2f}")
        else:
            st.write("Logistic Regression performance metrics not available.")

# หน้า 3

elif page == "Neural Network":
    st.title("Cat Breed Classification Using Neural Networks 🐱")

    st.header("Purpose")
    st.write(
        "This project aims to classify different cat breeds using a Convolutional Neural Network (CNN). "
        "By analyzing cat images, the model predicts the breed, providing a simple way to learn about various cat breeds."
    )

    st.header("Problem Type")
    st.write(
        "I approach this as a **multiclass image classification** problem, where each image is assigned "
        "to one of 67 cat breeds. The model uses deep learning to extract features from the images and differentiate between similar breeds."
    )

    st.header("Source Data")
    st.write(
        "The dataset for this project is obtained from Kaggle. "
        "You can access it here: [Cat Breeds Dataset](https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset?resource=download)"
    )
    st.markdown(
        """
        The dataset contains:
        - **Images of various cat breeds** with proper labeling.
        - **67 unique cat breeds** for classification.
        - **High-quality images** that are essential for effective model training.
        """
    )

    st.header("Data Preparation")
    st.write(
        "To prepare the data for training, I took several steps to ensure balance and quality:"
    )
    st.markdown(
        """
        1. **Balancing the Dataset**: Some breeds originally have up to 50,000 images, which can bias the model. "
        "I limited each breed to a maximum of **2,000 images** to ensure a balanced dataset.  
        2. **Data Augmentation**: I applied techniques such as rotation, shifting, zooming, and horizontal flipping to increase the variety of the training images.  
        3. **Train-Validation Split**: I split the balanced dataset into 80% training and 20% validation to monitor overfitting.
        """
    )
    st.write(
        "By capping each breed at 2,000 images, I prevent the model from becoming biased toward breeds with excessively large image counts."
    )

    st.header("Model")
    st.write(
        "I used a Convolutional Neural Network (CNN) that leverages transfer learning from a pretrained "
        "**MobileNetV2** model. The base model is frozen to preserve its learned features, and I added additional layers "
        "to specialize in cat breed classification."
    )
    st.markdown(
        """
        - **MobileNetV2**: A lightweight pretrained model effective for image feature extraction.  
        - **Fully Connected Layers**: Flatten the feature maps and apply dense layers for classification.  
        - **Batch Normalization & Dropout**: Improve training stability and reduce overfitting.  
        - **Softmax Activation**: Produces a probability distribution over 67 cat breeds.
        """
    )

    st.header("Model Performance")
    st.write("I monitored the model's performance using:")
    st.markdown(
        """
        - **Accuracy**: The percentage of correctly classified images.
        - **Loss**: Categorical cross-entropy loss indicating the prediction error.
        - **Validation Metrics**: To ensure the model generalizes well and adjust the learning rate when needed.
        """
    )

    st.header("Deployment")
    st.write(
        "After training and evaluating the CNN, I saved the model to `cat_breed_cnn.h5` and stored label information "
        "in `labels.json`. This enables the model to be deployed in a Streamlit application where users can upload a cat image "
        "and receive a breed prediction."
    )

    st.header("Neural Network Code")
    st.code(
        '''import os
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

# Parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 67
MAX_IMAGES_PER_CLASS = 2000

dataset_path = r"C:\\Users\\ppsuc\\OneDrive\\Desktop\\IS_project\\images"
balanced_path = r"C:\\Users\\ppsuc\\OneDrive\\Desktop\\IS_project\\balanced_images"

# Balance the dataset: limit each class to MAX_IMAGES_PER_CLASS images
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
                print(f"❌ Removing corrupted file: {img_src}")
                os.remove(img_src)

    print("✅ Balanced dataset created successfully!")

balance_dataset(dataset_path, balanced_path, MAX_IMAGES_PER_CLASS)

# Data Augmentation
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

# Compute class weights to address imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Use MobileNetV2 as the base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

# Build the CNN model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=min(len(train_data), 100),
    validation_steps=min(len(val_data), 50),
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler]
)

# Save the model and labels
model.save("cat_breed_cnn.h5")
print("✅ Model saved successfully!")

breed_labels = list(train_data.class_indices.keys())
labels_dict = {str(i): breed_labels[i] for i in range(len(breed_labels))}

with open("labels.json", "w") as f:
    json.dump(labels_dict, f)

print("✅ Labels saved successfully!")''',
        language='python'
    )



# หน้าที่ 4: อัปโหลดรูปเพื่อทดสอบโมเดล NN
elif page == "🐈 Guess the cat":
    st.title("Guess the cat by Neural Network")
    uploaded_file = st.file_uploader("Upload an image of your cat", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                st.image(uploaded_file, caption="Uploaded Image", width=400)
            except Exception as e:
                st.error(f"Cannot display the image. Error: {e}")

        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Confirm"):
                prediction = predict_cat_breed(uploaded_file)
                st.markdown(f"#### This is a cute **{prediction}**.")
