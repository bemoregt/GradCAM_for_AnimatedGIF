# GradCAM for Animated GIF

This repository contains a Python implementation of Gradient-weighted Class Activation Mapping (Grad-CAM) for animated GIF images. Grad-CAM is a visualization technique for deep learning networks that produces a visual explanation for decisions made by CNNs.

## Overview

This implementation applies Grad-CAM to each frame of an animated GIF, highlighting the regions that are most important for the model's prediction (in this case, detecting dogs in images). The result is a new animated GIF with heat maps overlaid on each frame.

## Features

- Process animated GIFs through a pre-trained ResNet-18 model
- Apply Grad-CAM visualization to each frame
- Overlay heatmaps on original frames
- Save the processed frames as a new animated GIF

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- NumPy
- OpenCV (cv2)
- imageio

## Usage

```python
# Set the paths for input and output GIFs
input_gif = "path/to/your/input.gif"
output_gif = "path/to/your/output.gif"

# Process the GIF
process_gif(input_gif, output_gif)
```

## How It Works

1. The code loads a pre-trained ResNet-18 model
2. Each frame of the input GIF is preprocessed and fed into the model
3. Grad-CAM is applied to visualize which regions of the image the model focuses on
4. A heatmap is generated and overlaid on the original frame
5. All processed frames are combined into a new GIF

## Details

The implementation defaults to ImageNet class index 208, which corresponds to 'dog' in the ImageNet classification. You can change this target class to visualize the model's focus for different object categories.

## Example

Input GIF | Output GIF with Grad-CAM
:--------:|:----------------------:
![Input GIF](example_input.gif) | ![Output GIF](example_output.gif)

Note: Example images are placeholders and need to be added separately.

## License

[MIT License](LICENSE)

## References

- Original Grad-CAM paper: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
