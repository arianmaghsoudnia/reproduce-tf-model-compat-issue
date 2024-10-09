TF_29_IMAGE = tf-model-training-2.9
TF_217_IMAGE = tf-model-training-2.17
VOLUME_PATH = .

build:
	docker build -t $(TF_29_IMAGE) -f Dockerfile-tf-2.9 .
	docker build -t $(TF_217_IMAGE) -f Dockerfile-tf-2.17 .

run:
	docker run --rm -v $(VOLUME_PATH)/stored_model_by_saved_model:/app/stored_model_by_saved_model -v $(VOLUME_PATH)/stored_model_by_keras:/app/stored_model_by_keras $(TF_29_IMAGE)
	docker run --rm -v $(VOLUME_PATH)/stored_model_by_saved_model:/app/stored_model_by_saved_model -v $(VOLUME_PATH)/stored_model_by_keras:/app/stored_model_by_keras $(TF_217_IMAGE)

clean:
	docker rmi $(TF_29_IMAGE) $(TF_217_IMAGE) || true

.PHONY: build run clean
