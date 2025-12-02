import os
from PIL import Image, ImageDraw

def create_dummy_data():
    data_dir = "../data"
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create a few dummy images
    image_names = ["1000268201_693b08cb0e.jpg", "1001773457_577c3a7d70.jpg"]
    colors = [(255, 0, 0), (0, 255, 0)]
    
    for name, color in zip(image_names, colors):
        path = os.path.join(images_dir, name)
        if not os.path.exists(path):
            img = Image.new('RGB', (224, 224), color=color)
            d = ImageDraw.Draw(img)
            d.text((10,10), "Dummy Image", fill=(255,255,255))
            img.save(path)
            print(f"Created dummy image: {path}")

if __name__ == "__main__":
    create_dummy_data()
