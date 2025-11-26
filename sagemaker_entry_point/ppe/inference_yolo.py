import os
import json
import torch
from ultralytics import YOLO
from PIL import Image
import io

def model_fn(model_dir):
    print(f"Loading the model: {model_dir}")
    model_path = os.path.join(model_dir, 'model.pt')
    
    if not os.path.exists(model_path):
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.pt'):
                    model_path = os.path.join(root, file)
                    break
    
    print(f"Model weights found: {model_path}")
    model = YOLO(model_path)
    return model

def _input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        return Image.open(io.BytesIO(request_body))
    else:
        raise ValueError(f"Tipo não suportado: {request_content_type}")

def _predict_fn(input_data, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    results = model.predict(input_data, conf=0.25)
    return results

def _output_fn(prediction):
    """Formata a saída do YOLO para JSON"""
    # O YOLO retorna uma lista de resultados (um por imagem)
    res = prediction[0] 
    
    output = []
    for box in res.boxes:
        # Extrai coordenadas, confiança e classe
        xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        cls_name = res.names[cls]
        
        output.append({
            'box': xyxy,
            'confidence': conf,
            'class_id': cls,
            'class_name': cls_name
        })
        
    return json.dumps(output)

def transform_fn(model, request_body, request_content_type, response_content_type):
    if request_content_type == 'application/x-image':
        input_data = _input_fn(request_body, request_content_type)
        predictions = _predict_fn(input_data, model)
        return _output_fn(predictions), response_content_type
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")