# 🫁 Chest X-Ray Pneumonia Detection AI

A deep learning model to detect pneumonia from chest X-rays using ResNet18 and PyTorch.

## Results
- Test Accuracy: 79.97%
- Trained on 5,216 chest X-rays
- Detects PNEUMONIA or NORMAL with confidence score

## Files
- pneumonia.py — Model training code
- app.py — Flask web app for live predictions

## How to Run
1. Install requirements: pip install torch torchvision flask pillow
2. Train the model: python pneumonia.py
3. Run the web app: python app.py
4. Open browser: http://127.0.0.1:5000

## Dataset
Kaggle Chest X-Ray Images (Pneumonia) by Paul Mooney

## Built by
Dr. Cyrus Yadav — MBBS | FMGE Qualified | AI in Healthcare
