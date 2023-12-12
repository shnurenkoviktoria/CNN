from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions

image_paths = [
    "dog.jpg",
    "car.jpg",
    "person.jpg",
    "bird.jpg",
    "cat.jpg",
    "shoe.jpg",
    "tshirt.jpg",
]
images = [img_to_array(load_img(path, target_size=(224, 224))) for path in image_paths]
images = [
    preprocess_input(image.reshape((1, image.shape[0], image.shape[1], image.shape[2])))
    for image in images
]

model_vgg = VGG16(include_top=True, weights="imagenet")

for i, image in enumerate(images):
    y_pred = model_vgg.predict(image)
    label = decode_predictions(y_pred)
    print(f"Image {i + 1} - {label[0][0][1]}: {label[0][0][2]:.4f}")
