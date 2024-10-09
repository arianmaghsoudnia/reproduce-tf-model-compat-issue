import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")


def create_keras_sequential_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(1),
        ]
    )
    return model


def generate_random_data(num_samples=1000):
    x = np.random.random((num_samples, 10))
    y = np.random.random((num_samples, 1))
    return x, y


def compile_model(model):
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])


def fit_model(model, x, y, epochs=10):
    model.fit(x, y, epochs=epochs)


def export_saved_model(model, export_dir="stored_model_by_saved_model"):
    tf.saved_model.save(model, export_dir)


def load_saved_model(export_dir="stored_model_by_saved_model"):
    return tf.saved_model.load(export_dir)


def export_keras_model(model, export_dir="stored_model_by_keras"):
    tf.keras.models.save_model(model, export_dir)


def load_keras_model(export_dir="stored_model_by_keras"):
    return tf.keras.models.load_model(export_dir)


def create_compile_fit_save_model():
    model = create_keras_sequential_model()
    compile_model(model)
    x, y = generate_random_data()
    fit_model(model, x, y)
    export_saved_model(model)
    export_keras_model(model)


def load_predict_and_assert():
    loaded_model_saved_model = None
    loaded_model_keras = None

    try:
        loaded_model_saved_model = load_saved_model()
    except Exception as e:
        print(
            f"Model which was saved by older tf module tf.saved_model.save, failed to load with newer tf module tf.saved_model.load : {e}"
        )

    try:
        loaded_model_keras = load_keras_model()
    except Exception as e:
        print(
            f"Model which was saved by older Keras (v2) module keras.models.save_model, failed to load with newer Keras module tf.keras.models.load_model : {e}"
        )

    if loaded_model_saved_model is not None and loaded_model_keras is not None:
        random_input = np.random.random((1, 10)).astype(np.float32)
        predictions_saved_model = loaded_model_saved_model(random_input)
        predictions_keras_model = loaded_model_keras.predict(random_input)
        print("Predictions from loaded saved model:", predictions_saved_model.numpy())
        print("Predictions from loaded Keras model:", predictions_keras_model)
        assert np.allclose(
            predictions_saved_model.numpy(), predictions_keras_model
        ), "Predictions from both models do not match!"
    else:
        print(
            "One of the models could not be loaded and thus no assertion can be made!"
        )


def main():
    create_compile_fit_save_model()
    load_predict_and_assert()


if __name__ == "__main__":
    main()
