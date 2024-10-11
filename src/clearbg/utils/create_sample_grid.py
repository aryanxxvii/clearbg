import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from clearbg.constants import PROJECT_ROOT

# Directory containing the images
image_dir = PROJECT_ROOT / '_resources' / '_samples'
save_path = PROJECT_ROOT / '_resources' / 'sample_grid.png'

# Get all image files in the directory
all_files = list(image_dir.glob('*.png'))




# Define the types of images and their titles
image_types = ['org', 'seg', 'tra', 'ans']
image_titles = ['Input Image', 'Output Segmentation Mask', 'Transparent Background', 'Ground Truth Mask']

num_columns = len(image_types)  # Number of columns
# Calculate the number of image sets
num_images = len(all_files) // num_columns

# Create a list to hold the image file names
image_files = []

# Generate the image file names based on the naming convention
for i in range(1, num_images + 1):
    image_row = [image_dir / f'p{i}_{img_type}.png' for img_type in image_types]
    image_files.append(image_row)




# Create a figure for the grid (num_images rows, num_columns columns)
num_columns = len(image_types)  # Number of columns
individual_width = 3            # Width for each image
individual_height = 3           # Height for each image

# Create the figure with dynamic size
fig, axs = plt.subplots(num_images, num_columns, figsize=(individual_width * num_columns, individual_height * num_images))

for j, title in enumerate(image_titles):
    axs[0, j].set_title(title, fontsize=16)

# Customize the title for the Output Segmentation Mask
axs[0, 1].set_title('', fontsize=16)  # Clear the existing title for custom text

# Add custom colored text for Output Segmentation Mask
axs[0, 1].text(0.4, 1.02, 'Output', ha='center', va='bottom', fontsize=16, color='blue', transform=axs[0, 1].transAxes)
axs[0, 1].text(0.65, 1.02, ' Mask', ha='center', va='bottom', fontsize=16, color='red', transform=axs[0, 1].transAxes)


# Populate the grid with images
for i in range(num_images):
    for j in range(num_columns):
        img_path = image_files[i][j]
        
        # Check if the image file exists
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB mode
            
            # Stretch the image to fit the axes
            axs[i, j].imshow(image, aspect='auto')  # Use aspect='auto' to stretch the image
        else:
            axs[i, j].imshow([[0, 0, 0]])  # Show a black placeholder if the file is missing
            axs[i, j].set_title(f'Missing: {img_path.name}', fontsize=14)  # Indicate missing file

        axs[i, j].axis('off')  # Hide axis

# Add a grid to the background for visual separation
for ax in axs.flatten():
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

# Adjust spacing manually for better layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.02, hspace=0.02)

# Save the final grid image
plt.savefig(save_path, bbox_inches='tight')  # Save with tight bounding box

