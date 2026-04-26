# Predict a new flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Predicted class:", iris.target_names[prediction][0])