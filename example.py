from gradcam_for_gif import process_gif

# Example usage with different target classes
# See https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a for ImageNet class indices
# 208: Labrador retriever
# 209: Golden retriever
# 285: Egyptian cat
# 287: Lynx

# Process GIF looking for dogs (class 208 - Labrador retriever)
process_gif(
    input_gif="input.gif",  # Replace with your input GIF path
    output_gif="output_dog.gif",
    target_class=208
)

# Process same GIF looking for cats (class 285 - Egyptian cat)
process_gif(
    input_gif="input.gif",  # Replace with your input GIF path
    output_gif="output_cat.gif",
    target_class=285
)
