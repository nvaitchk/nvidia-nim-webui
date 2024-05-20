# Web UI for Stable Diffusion XL NIM (On-Premises Deployment)
![image](https://github.com/nvaitchk/nvidia-nim-webui/assets/89618410/ae78c8fd-58f6-4bcf-86be-b4e6b6ab0fc2)

## Overview

This project demonstrates the power and simplicity of NVIDIA NIM (NVIDIA Inference Microservice), a suite of optimized cloud-native microservices, by setting up and running a Stable Diffusion XL pipeline. NVIDIA NIM is designed to streamline the deployment and time-to-market of generative AI models across various environments, including cloud platforms, data centers, and GPU-accelerated workstations. By abstracting the complexities of AI model development and leveraging industry-standard APIs, NIM makes advanced AI technologies accessible to a broader range of developers.

## Getting Started

### Prerequisites

- SDXL NIM deployed on NVIDIA GPU(s)
- Python 3.10 or higher
- Anaconda

### Installation

1. **Clone the repository**

    ```
    git clone https://github.com/nvaitchk/nvidia-nim-workshops.git
    cd NIM_SDXL
    ```

2. **Set up a conda environment**

    Create a conda environment named `nim_sdxl`:

    ```
    conda create -n nim_sdxl python=3.10
    ```

    Activate the conda environment:

    ```
    conda activate nim_sdxl
    ```

3. **Install dependencies**

    Install the required packages using pip:

    ```
    pip install -r requirements.txt
    ```

### Usage

To run the project, execute the following command:

```
python SDXL_NIM_UI.py
```
