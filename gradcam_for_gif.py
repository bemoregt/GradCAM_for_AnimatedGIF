import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from imageio import mimread, mimsave

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        target = model_output[0, target_class]
        target.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def overlay_heatmap(image, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return superimposed_img

def process_gif(input_gif, output_gif, target_class=208):  # 208 is the index for 'dog' in ImageNet
    """
    Apply Grad-CAM visualization to each frame of an animated GIF.
    
    Args:
        input_gif: Path to the input GIF file
        output_gif: Path to save the output GIF file
        target_class: ImageNet class index to visualize (default: 208 for 'dog')
    """
    # Load pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Create GradCAM object
    grad_cam = GradCAM(model, model.layer4[-1])
    
    # Load GIF frames
    frames = mimread(input_gif)
    
    # Process each frame
    processed_frames = []
    for frame in frames:
        pil_image = Image.fromarray(frame).convert('RGB')
        input_tensor = preprocess_image(pil_image)
        
        # Get Grad-CAM heatmap
        cam = grad_cam.generate_cam(input_tensor, target_class)
        
        # Resize frame to match the CAM size
        frame_resized = cv2.resize(np.array(pil_image), (224, 224))
        
        # Overlay heatmap on original frame
        overlayed_frame = overlay_heatmap(frame_resized, cam)
        processed_frames.append(overlayed_frame)
    
    # Save processed frames as new GIF
    mimsave(output_gif, processed_frames, duration=100)  # Adjust duration as needed

# Example usage
if __name__ == "__main__":
    input_gif = "input.gif"  # Replace with your input GIF path
    output_gif = "output.gif"  # Replace with your output GIF path
    
    process_gif(input_gif, output_gif)
