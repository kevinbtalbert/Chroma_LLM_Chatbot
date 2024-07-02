# Chroma LLM Chatbot

## Setup Instructions

### Verify CUDA 
Login to the target device and run: `nvidia-smi` and verify an output such as below:

```
[ec2-user@ip-10-10-2-109 HOL_FILES]$ nvidia-smi
Tue Jul  2 23:05:30 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   32C    P8               9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## Install

1. `git clone https://github.com/kevinbtalbert/Chroma_LLM_Chatbot.git`

2. Using local Python installation (e.g. Python 3.11) and reference it by its alias (e.g. `python3.11`) execute:

Confirm version/alias with `python --version` or `python3.11 --version`

`python -m hol_venv venv` or with alias `python3.11 -m hol_venv venv`

3. With virtual environment activated, run `source hol_venv/bin/activate` and initialize virtual environment.

4. From virtual environment, run `pip install -r requirements.txt` from within the git directory downloaded.

5. Run `python setup-chromadb.py` from within the git directory downloaded.

6. Run pip uninstall / install pysqlite3-binary like `pip install pysqlite3-binary` for chroma dependency.

7. Set `HF_ACCESS_TOKEN` in `config.ini` with a Hugging Face API Key with at minimum Read permissions to the `mistralai/Mistral-7B-Instruct-v0.2` LLM model.

## Launch Application

Run `python llm-app.py` to launch the application from within the git directory downloaded. Adjust any necessary configs in `config.ini` file in project directory.


Verify CUDA consumption by model:

```
[ec2-user@ip-10-10-2-109 HOL_FILES]$ nvidia-smi
Tue Jul  2 23:23:51 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   39C    P0              33W /  70W |   4077MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      4672      C   python                                     4072MiB |
+---------------------------------------------------------------------------------------+
```
