from typing import Literal, Tuple
import tensorflow as tf
import numpy as np


def log_tf_version():
    print(f"TF version: {tf.__version__}")


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


def load_model(export_dir: str, method: Literal["saved_model", "keras"]):
    if method == "saved_model":
        return tf.saved_model.load(export_dir)
    elif method == "keras":
        return tf.keras.models.load_model(export_dir)
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
    models = {"saved_model": None, "keras": None}

    for method in models.keys():
        try:
            models[method] = load_model(f"stored_model_by_{method}", method=method)
        except Exception as e:
            print(f"Failed to load {method} model: {e}")
    if all(models.values()):
        random_input = np.random.random((1, 10)).astype(np.float32)
        predictions = {
            "saved_model": models["saved_model"](random_input),  # pylint: disable=E1102
            "keras": models["keras"].predict(random_input),
        }
        print(
            "Predictions from loaded saved model:", predictions["saved_model"].numpy()
        )
        print("Predictions from loaded Keras model:", predictions["keras"])
        assert np.allclose(
            predictions["saved_model"].numpy(), predictions["keras"]
        ), "Predictions do not match!"
    else:
        print("One of the models could not be loaded, no assertion can be made.")


def main() -> None:
    log_tf_version()
    create_compile_fit_save_model()
    load_predict_and_assert()


if __name__ == "__main__":
    main()
