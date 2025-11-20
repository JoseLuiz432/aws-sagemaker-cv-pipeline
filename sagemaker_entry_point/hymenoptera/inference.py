import torch
import os
import io
from torchvision import models, transforms
from PIL import Image
from data import get_transforms
import json
from model import load_model_resnet18

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Carregando modelo no device: {device}")
    
    model = load_model_resnet18(pretrained=False)
    
    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval() # Modo de avaliação (trava dropout/batchnorm)
    return model

def _input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body))
        
        preprocess = get_transforms(is_training=False)
        return preprocess(image).unsqueeze(0)
    
    raise ValueError(f"Tipo de conteúdo não suportado: {request_content_type}")

def _predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    
    with torch.no_grad():
        return model(input_data)
    
def transform_fn(model, request_body, request_content_type, response_content_type):
    if request_content_type == 'application/x-image':
        input_data = _input_fn(request_body, request_content_type)
        outputs = _predict_fn(input_data, model)
        
        return json.dumps(outputs.cpu().numpy().tolist()), response_content_type