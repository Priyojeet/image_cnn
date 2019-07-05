import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from matplotlib.image import imread




# create a list according to your data_directory and the order must be same.
column = ["jeet's", "ratna's", "soumya's", "subhajit's", "sudhanshu's"]
new_model = load_model('model1.h5')
new_model.summary()
test_image = image.load_img("path of your img file you want to predict.", target_size = (400, 400, 3))
test_image = image.img_to_array(test_image)
test_image = test_image/225
plt.imshow(test_image)
result = new_model.predict(test_image.reshape(1, 400, 400, 3))
print(result)
print(result[0][0])
# get the prediction in boolean
pred_bool = (result >0.5)
print(pred_bool)
# predictions in int
predictions = pred_bool.astype(int)
#print(predictions)
for i in predictions:
    d = dict(zip(column, i))
final_prediction = [key  for (key, value) in d.items() if value == 1]
#getting the final value
print(final_prediction)

