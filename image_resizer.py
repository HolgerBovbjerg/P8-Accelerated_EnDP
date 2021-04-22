#Import required Image library
from PIL import Image
import os

directory = os.fsencode('/Code/Uni/P8-Accelerated_EnDP/Raw_images')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".JPEG"):
        # Create an Image Object from an Image
        im = Image.open(f'/Code/Uni/P8-Accelerated_EnDP/Raw_images/{filename}')

        # Display actual image
        # im.show()

        # Make the new image half the width and half the height of the original image
        resized_im = im.resize((224, 224))

        # Display the resized imaged
        # resized_im.show()

        # Save the cropped image
        resized_im.save(f'/Code/Uni/P8-Accelerated_EnDP/Resized_images/resized_{filename}')
    else:
        pass

