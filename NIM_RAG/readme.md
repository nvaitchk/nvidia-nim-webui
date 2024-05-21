# Web UI for RAG using NVIDIA NIM with LangChain

![image](https://github.com/nvaitchk/nvidia-nim-webui/assets/89618410/04713976-7dd2-459b-b5c1-ed9bd896a7b1)


## Overview

This project demonstrates the power and simplicity of NVIDIA NIM (NVIDIA Inference Microservice), a suite of optimized cloud-native microservices, by setting up and running a Retrieval-Augmented Generation (RAG) pipeline. NVIDIA NIM is designed to streamline the deployment and time-to-market of generative AI models across various environments, including cloud platforms, data centers, and GPU-accelerated workstations. By abstracting the complexities of AI model development and leveraging industry-standard APIs, NIM makes advanced AI technologies accessible to a broader range of developers.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Anaconda
- API key from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nemo-8b-steerlm

### Installation

1. **Clone the repository**

    ```
    git clone https://github.com/nvaitchk/nvidia-nim-webui.git
    cd NIM_RAG
    ```

2. **Set up a conda environment**

    Create a conda environment named `nim_rag`:

    ```
    conda create -n nim_rag python=3.10
    ```

    Activate the conda environment:

    ```
    conda activate nim_rag
    ```

3. **Install dependencies**

    Install the required packages using pip:

    ```
    pip install -r requirements.txt
    ```

4. **Environment Variables**

    Set your NVIDIA API key as an environment variable for the conda environment:

    ```
    conda env config vars set NVIDIA_API_KEY=your_nvidia_api_key_here
    ```

    Replace `your_nvidia_api_key_here` with your actual NVIDIA API key.

4. **Reactivate the conda environment**

    ```
    conda deactivate 
    conda activate nim_rag
    ```


### Usage

To run the project, execute the following command:

```
python RAG_UI.py
```
