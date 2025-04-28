# Description
This project is based on SgLang v0.4.3. It adds compression algorithms to accelerate the performance of KV inference. 

# Installation
For this project, please install it using "Install from source". Here are the commands:
```bash
pip install --upgrade pip
pip install sgl-kernel --force-reinstall --no-deps
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer-python
```

# Running the project
In ths root directory, there is a jupyter notebook named `test.ipynb`. You can specify the model, test data, and save path in the `Config` class. After running the notebook, you can check the output in the `config.save_path`.

# Usage
This project supports compression while the program is decoding. The default compression method is named "newline". In order to use this method, you need to specify the following parameters in the script:
* compress_algorithm="CustomKV"
* compress_max_window with an integer. For example, 8. 
* compress_max_prompt with an integer. For example, 128. 
* compress_divide_method="newline"