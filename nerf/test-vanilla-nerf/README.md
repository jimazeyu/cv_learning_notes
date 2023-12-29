# Test NeRF
Build vanilla nerf with the template.

## Annotations
1. The file pyproject.toml is to help install, leading to my_config.py.
2. The file my_config.py is to help registration. It imports other files and submmit to nerf studio.
3. The data manager defines data format and transmits data.
4. The model defines model.
5. The field generates outputs.
4. The pipeline file compose the data manager, the model and the field. It finally calculate the loss dict.


# Template Annotations

## File Structure
We recommend the following file structure:

```
├── my_method
│   ├── __init__.py
│   ├── my_config.py
│   ├── custom_pipeline.py [optional]
│   ├── custom_model.py [optional]
│   ├── custom_field.py [optional]
│   ├── custom_datamanger.py [optional]
│   ├── custom_dataparser.py [optional]
│   ├── ...
├── pyproject.toml
```

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "method-template". To train with it, run the command:
```
ns-train method-template --data [PATH]
```