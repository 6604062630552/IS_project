import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train_and_save_metrics(noise_std=0.5):
    # โหลดข้อมูล
    dataset_file = "house.csv"
    df = pd.read_csv(dataset_file)

    # กำหนดฟีเจอร์ที่เป็นตัวเลข
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
    
    # สร้าง DataFrame สำเนาเพื่อลง Noise
    df_noisy = df.copy()

    # เพิ่ม Gaussian Noise ให้กับแต่ละฟีเจอร์
    for col in features:
        noise = np.random.normal(0, noise_std, size=df_noisy.shape[0])
        # หากต้องการคงเป็น float ในช่วง 1-10
        df_noisy[col] = (df_noisy[col] + noise).clip(lower=1, upper=10)

    X = df_noisy[features]
    y = df_noisy["House"]

    # สุ่มเรียงข้อมูลใหม่
    df_noisy = df_noisy.sample(frac=1, random_state=None).reset_index(drop=True)

    # แบ่ง Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # เทรนโมเดล Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
    joblib.dump(rf_model, "house_rf.pkl")

    # เทรนโมเดล Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')
    joblib.dump(lr_model, "house_lr.pkl")

    # เก็บค่าความแม่นยำใน st.session_state
    if "rf_accuracy" not in st.session_state:
        st.session_state["rf_accuracy"] = rf_accuracy
        st.session_state["rf_f1"] = rf_f1
        st.session_state["lr_accuracy"] = lr_accuracy
        st.session_state["lr_f1"] = lr_f1


if __name__ == "__main__":
    # เรียกใช้ฟังก์ชันเทรนและบันทึกโมเดล พร้อม Noise ที่กำหนด
    train_and_save_metrics(noise_std=2.0)
