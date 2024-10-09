from typing import Literal, Tuple
import tensorflow as tf
import numpy as np


def log_tf_version():
    print(f"TF version: {tf.__version__}")


log_tf_version()


def create_keras_sequential_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(1),
        ]
    )


def generate_random_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.random((num_samples, 10))
    y = np.random.random((num_samples, 1))
    return x, y


def compile_model(model: tf.keras.Model) -> None:
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])


def fit_model(
    model: tf.keras.Model, x: np.ndarray, y: np.ndarray, epochs: int = 10
) -> None:
    model.fit(x, y, epochs=epochs)


def export_model(
    model: tf.keras.Model, export_dir: str, method: Literal["saved_model", "keras"]
) -> None:
    if method == "saved_model":
        tf.saved_model.save(model, export_dir)
    elif method == "keras":
        tf.keras.models.save_model(model, export_dir)
    else:
        raise ValueError("Unknown export method specified.")


def load_model(
    export_dir: str,
    method: Literal["saved_model", "keras"],
    as_tfsm_layer: bool = False,
):
    if method == "saved_model":
        return tf.saved_model.load(export_dir)
    elif method == "keras":
        if as_tfsm_layer:
            return tf.keras.layers.TFSMLayer(
                export_dir, call_endpoint="serving_default"
            )
        else:
            return tf.keras.models.load_model(export_dir, compile=False)
    else:
        raise ValueError("Unknown load method specified.")


def create_compile_fit_save_model() -> None:
    model = create_keras_sequential_model()
    compile_model(model)
    x, y = generate_random_data()
    fit_model(model, x, y)
    export_model(model, "stored_model_by_saved_model", method="saved_model")
    export_model(model, "stored_model_by_keras", method="keras")


def load_predict_and_assert() -> None:
    all_models_can_be_loaded = True
    random_input = np.random.random((1, 10)).astype(np.float32)
    for method in ["saved_model_default", "keras_tfsm", "keras"]:
        predictions = []
        try:
            if method == "saved_model_default":
                saved_model = load_model(
                    "stored_model_by_saved_model", method="saved_model"
                )
                prediction_saved_model = saved_model(random_input).numpy()
                print(
                    "Predictions from loaded saved model:",
                    prediction_saved_model,
                )
                predictions.append(prediction_saved_model)
            elif method == "keras_tfsm":
                tfsm_model = load_model(
                    "stored_model_by_keras", method="keras", as_tfsm_layer=True
                )
                prediction_keras_model_tfsm = tfsm_model(random_input)[
                    "dense_1"
                ].numpy()
                print(
                    "Predictions from loaded Keras model (loaded as TFSM) :",
                    prediction_keras_model_tfsm,
                )
                predictions.append(prediction_keras_model_tfsm)
            else:
                keras_model = load_model(
                    "stored_model_by_keras", method="keras", as_tfsm_layer=False
                )
                prediction_keras_model_direct = keras_model.predict(
                    random_input
                ).ravel()
                print(
                    "Predictions from loaded Keras model (loaded directly):",
                    prediction_keras_model_direct,
                )
                predictions.append(prediction_keras_model_direct)
        except Exception as e:
            print(f"Failed to load {method} model: {e}")
            all_models_can_be_loaded = False

    if all_models_can_be_loaded:
        assert np.allclose(np.array(predictions), "Predictions do not match!")
    else:
        print("One of the models could not be loaded, no assertion can be made.")


def main() -> None:
    log_tf_version()
    create_compile_fit_save_model()
    load_predict_and_assert()


if __name__ == "__main__":
    main()
