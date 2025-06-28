 
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

# Use raw string (prefix with r) or os.path.join for proper path formatting
train_dir = os.path.join('output_dataset', 'train')

generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("Class Indices:", generator.class_indices)








