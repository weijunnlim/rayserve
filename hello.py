import requests
from starlette.requests import Request
from typing import Dict
import torch
import io
from transformers import pipeline
from PIL import Image
from torchvision import transforms
from ray import serve


@serve.deployment
class CatDogClassifier:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load("/home/dxd_wj/model_serving/rayserve/model.pth", map_location=device) #depends if gpu is available or not
        self.model.eval() 

        # Define the image transformation pipeline  
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    async def __call__(self, request: Request) -> Dict:
        image_bytes = await request.body()
        image = Image.open(io.BytesIO(image_bytes))

        image = self.transform(image).unsqueeze(0) 

        #Run the model on the input image
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            label = "Cat" if predicted.item() == 0 else "Dog" # 0 if cat, 1 if dog

        return {"label": label}


# 2: Deploy the deployment.
serve.run(CatDogClassifier.bind(), route_prefix="/predict")

# 3: Query the deployment and print the result.
image_path = "/home/dxd_wj/model_serving/images/rayserve/image6.jpg"
with open(image_path, "rb") as f:
    image_bytes = f.read()

response = requests.post("http://localhost:8000/predict", data=image_bytes)
print(response.json())