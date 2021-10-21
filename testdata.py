
import matplotlib.pyplot as plt
# grab some test images from the test data
test_images = testData[1:5]
# reshape the test images to standard 28x28 format
test_images = test_images.reshape(test_images.shape[0], 28, 28)
print "[INFO] test images shape - {}".format(test_images.shape)
# loop over each of the test images
for i, test_image in enumerate(test_images, start=1):
# grab a copy of test image for viewing
org_image = test_image
# reshape the test image to [1x784] format so that our model understands
test_image = test_image.reshape(1,784)
# make prediction on test image using our trained model
prediction = model.predict_classes(test_image, verbose=0)
# display the prediction and image
print "[INFO] I think the digit is - {}".format(prediction[0])
plt.subplot(220+i)
plt.imshow(org_image, cmap=plt.get_cmap('gray'))
plt.show()
