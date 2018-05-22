
import os

def save_model(model,model_name='model.json',weights_name='model.h5'):
    model_json = model.to_json()
    with open(model_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_name)
    print("Saved model to disk")

