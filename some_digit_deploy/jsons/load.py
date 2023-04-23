from keras.models import model_from_json


def init(json_path: str, weights_path: str):
    with open(json_path, 'r') as arquivo_json:
        modelo_carregado = model_from_json(arquivo_json.read())
        modelo_carregado.load_weights(weights_path)

    return modelo_carregado
