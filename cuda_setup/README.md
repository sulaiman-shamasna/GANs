# CUDA Setup Guideline
---
This guide aims at leading you to prepare a *Tensorflow* training environment with *GPU* on Windows machine.

1. Download and install [NVIDIA Driver](https://www.nvidia.com/en-us/drivers/)
2. Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows)
3. Download and install [Anaconda](https://www.anaconda.com/download).
4. Either during *Anaconda* installation or afterwards, don't forget to add the *Anaconda* to the system path. This is done by opening the *Edit the system environment variables* window, then follow the plots shown bellow:
![Add Anaconda environment variable path](https://github.com/sulaiman-shamasna/GANs/blob/main/cuda_setup/plots/env_variable.png)
5. Create a new conda environment with *Python 3.10*, using the command.
    ```bash
    conda create -n py310 python=3.10
    ```
6. and activate it, using the command:
    ```bash
    conda activate py310
    ```
7. Install cuda-related packages, e.g.,:
    ```bash
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    ```
8. Install *Tensorflow 2.10*, using the command:
    ```bash
    python -m pip install "tensorflow==2.10"
    ```
9. Test successful installations. Create a simple python script, type the following inside and run it, the output is supposed to be *True*:
    ```python
    import tensorflow as tf
    # tf.config.list_physical_devices('GPU')
    print('--->', tf.test.is_gpu_available())

    """OUTPUT

    ---> True
    """
    ```

***Working with Jupyter Notebook***
In case you're working with a jupyter notebook, follow the following settings **after creating and activiting the conda environment**:
1. *Install Jupyter in the Conda Environment*: After activating the environment, install Jupyter Notebook:
    ```bash
    conda install -c conda-forge notebook
    ```
2. *Install the ```ipykernel``` Package*: To make the environment available in Jupyter as a kernel, install ipykernel:
    ```bash
    conda install ipykernel
    ```
3. *Add the Environment as a Jupyter Kernel*: Now, add your environment as an option in Jupyter:
    ```bash
    python -m ipykernel install --user --name=myenv --display-name "Python (REPLACE_WITH_ENV_NAME)"
    ```
4. *Open Jupyter Notebook*: Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```