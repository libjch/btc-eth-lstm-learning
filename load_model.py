from keras.models import model_from_json


def load_model(model_name='model.json',weights_name='model.h5'):
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_name)
    print("Loaded model from disk")
    loaded_model.compile(loss='mse', optimizer='adam')
    return loaded_model
