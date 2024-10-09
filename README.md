
# reproduce-tf-model-compat-issue

This repository provides a minimal example to reproduce a backward compatibility issue when loading TensorFlow saved models across different versions of TensorFlow.

## Overview

The code demonstrates how to:

1. Create a simple Keras sequential model.
2. Train the model with random data.
3. Export the trained model using two different methods: TensorFlow's `saved_model` format and `Keras` model format.
4. Load the models in two different TensorFlow versions (2.9 and 2.17) to check for compatibility issues.
5. Compare the predictions from both loaded models to assert they match.

## Files

- `model_training_export_and_load.py`: Contains the code for model creation, training, exporting, loading, and prediction.
- `Makefile`: Defines the build and run commands for Docker containers with different TensorFlow versions.

## Usage

### Prerequisites

Make sure you have Docker installed on your machine.

### Build Docker Images

To build the Docker images for TensorFlow versions 2.9 and 2.17, run:

```sh
make build
```

### Run the Models

To run the model training and loading:

```sh
make run
```

This step uses TensorFlow version 2.9 to train the model on random data and save it and uses TensorFlow versions 2.17 to load the saved model and do the prediction on a random input.

### Clean Up

To remove the built Docker images, run:

```sh
make clean
```

## Conclusion

Loading saved models created with Keras 2 in a newer version of TensorFlow that uses the Keras 3 API consistently fails. Attempting to load such models results in the following errors:

- Fails to load using tf.saved_model.load: '_UserObject' object has no attribute 'add_slot'
- Fails to load using keras.layers.TFSMLayer: '_UserObject' object has no attribute 'add_slot'
- Fails to load using tf.keras.models.load_model: File format not supported.
  
This demonstrates a clear backward compatibility issue: models saved with Keras 2 cannot be loaded using the same method in Keras 3. This limitation particularly arises because Keras 3 only supports V3 .keras files and legacy .h5 format, while the legacy "tf" format is not supported.

Even when utilizing the SavedModel bundle in the newer TensorFlow version, the following error can occur: AttributeError: '_UserObject' object has no attribute 'add_slot'. The suggested workaround of using keras.layers.TFSMLayer does not resolve the issue, as it relies on tf.saved_model.load, which also encounters the same limitation.

Encounters with this challenge suggest that users may need to rebuild their models for compatibility with newer TensorFlow versions, as community feedback has indicated a lack of immediate solutions for seamless loading.



