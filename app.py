from flask import Flask, request, jsonify, render_template_string
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

app = Flask(__name__)

model = models.resnet18()
model.fc = nn.Linear(512, 1)
model.load_state_dict(torch.load(r'C:\Users\User\pneumonia_model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Pneumonia Detector</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 100px auto; text-align: center; background: #f0f4f8; }
        h1 { color: #2c3e50; }
        input { margin: 20px; }
        button { background: #2980b9; color: white; padding: 10px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        #result { margin-top: 30px; font-size: 24px; font-weight: bold; }
        .pneumonia { color: red; }
        .normal { color: green; }
    </style>
</head>
<body>
    <h1>🫁 Chest X-Ray Pneumonia Detector</h1>
    <p>Upload a chest X-ray image to analyze</p>
    <input type="file" id="fileInput" accept="image/*"><br>
    <button onclick="analyze()">Analyze X-Ray</button>
    <div id="result"></div>
    <script>
        async function analyze() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) { alert('Please select an image first'); return; }
            const formData = new FormData();
            formData.append('file', file);
            document.getElementById('result').innerHTML = 'Analyzing...';
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            const resultDiv = document.getElementById('result');
            if (data.prediction === 'PNEUMONIA') {
                resultDiv.innerHTML = '⚠️ PNEUMONIA Detected (' + data.confidence + '% confidence)';
                resultDiv.className = 'pneumonia';
            } else {
                resultDiv.innerHTML = '✅ NORMAL (' + data.confidence + '% confidence)';
                resultDiv.className = 'normal';
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor).squeeze()
        prob = torch.sigmoid(output).item()
    prediction = 'PNEUMONIA' if prob > 0.5 else 'NORMAL'
    confidence = round(prob * 100 if prob > 0.5 else (1 - prob) * 100, 2)
    return jsonify({'prediction': prediction, 'confidence': confidence})

if name == '__main__':
    app.run(debug=True)
