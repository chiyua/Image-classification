import sys
import numpy as np
from os import path

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def load_image(filename):
	image = load_img(filename, target_size=(32, 32))
	image = img_to_array(image)
	image = image.reshape(1, 32, 32, 3)
	image = image.astype('float32')/255.0
	return image

def predict():
	try:
		image_path = 'airplane.jpg'
	except:
		print('No path was passed in the argument.')
		exit()
	
	if (not path.exists(image_path)):
		print('Image file does not exist.')
		exit()

	image = load_image(image_path)
	model = load_model('model/my_model_two.keras')
	result = model.predict(image)
	# Get the predicted class (assuming a classification task)
	predicted_class = np.argmax(result[0])
	print("Predicted class:", predicted_class)
	plt.imshow(mpimg.imread(image_path))
	plt.axis('off')
	plt.title('Predicted label: {}'.format(labels[predicted_class]))
	plt.show()
 
predict()