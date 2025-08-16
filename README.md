![Multiple Disease Prediction System](https://0701.static.prezi.com/preview/v2/xdiuntc2v22m7frjzpyffrbje36jc3sachvcdoaizecfr3dnitcq_3_0.png)

# 🧠 Multi-Disease Prediction Web App

This project is a **Streamlit-based web application** that uses **Artificial Neural Networks (ANNs)** to predict the risk of multiple diseases:
- ❤️ Heart Attack
- 🎀 Breast Cancer
- 🍬 Diabetes

The user can select which disease to check from the **sidebar menu**, input the required health parameters, and get an instant prediction.

---

## 🚀 Features
- **Multi-task Support**: Heart Attack, Breast Cancer, and Diabetes prediction in a single app.
- **Interactive UI**: Built with Streamlit for simplicity and usability.
- **Deep Learning Models**: Trained ANN models using TensorFlow/Keras.
- **Scalable Design**: Easily extendable to add more diseases in the future.

---

## 📂 Project Structure

Multi-Disease-Prediction/
│── app.py
│── breast_cancer_ann_model.keras
│── brstcancer_scaler.pkl
│── diabetes_ann_model.keras
│── diabetes_scaler.pkl
│── HAP Model.H5
│── HAPM_standardScaler.pkl
│── License
│── requirements

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Multi-Disease-Prediction.git
cd Multi-Disease-Prediction
```

## 🐍 Create virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate

## 📦 Install dependencies

```bash
pip install -r requirements.txt
```

## 🌐 Run the Streamlit app
```bash
streamlit run app.py
```

## 📊 Models:

- Heart Attack Prediction: ANN trained on heart disease dataset.
- Diabetes Prediction: ANN trained on PIMA Indians diabetes dataset.
- Breast Cancer Prediction: ANN trained on UCI Breast Cancer dataset.
- All models were saved using Keras (.keras format) and use a shared scaler (scaler.pkl) for preprocessing.

## 🤝 Contributing

- Pull requests are welcome. If you’d like to add another disease model, simply:
- Train the ANN model.
- Save it in the models/ folder.
- Update app.py with the new option.

## 📜 License

This project is licensed under the MIT License.
