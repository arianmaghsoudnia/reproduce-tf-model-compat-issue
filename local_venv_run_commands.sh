# TF 2.17
python3.11 -m venv venv_tf_2_17
source ./venv_tf_2_17/bin/activate
pip install -r requirements_tf_2_17.txt
python -c "import model_training_export_and_load; model_training_export_and_load.load_predict_and_assert()"

# TF 2.9
python3.8 -m venv venv_tf_2_9
source ./venv_tf_2_9/bin/activate
pip install -r requirements_tf_2_9.txt
python -c "import model_training_export_and_load; model_training_export_and_load.create_compile_fit_save_model()"
