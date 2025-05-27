# Exploring Subtle Manipulation Vulnerabilities in Federated Distillation
## Introduction
![Uploading image.png…]()

## Dependencies
* Python 3.8  
* PyTorch (`torch`)  2.7.0  
* PyTorch (`torchvision`) 0.22.0
* wandb 0.19.11
* hnswlib 0.8.0
* matplotlib 3.10.3
* Scikit-learn (`scikit-learn`)  1.6.1
* NumPy (`numpy`)  2.1.3

## Installation and Configuration
### 1. Create and activate a Conda environment
**Create a Conda environment named 'peiy' (ensure Python version is compatible with the project).**

```
conda create -n peiy python=3.8 -y
```
**Activate the environment**
```
conda activate peiy
```

### 2. Install dependencies
 **Install PyTorch Dependencies**
```
pip install torch torchvision
```
**Install Additional Dependencies**
```
pip install matplotlib          # 绘图库 
pip install scikit-learn        # 机器学习工具
pip install hnswlib             # 高效向量搜索库
pip install wandb               # 实验跟踪工具
```
## File Structure
```
ReCServe/
├── main.py               # Main script for running experiments
├── recursive_serve.py    # Core implementation of the ReCServe framework
├── base_serve.py         # Basic serving class
├── evaluation.py         # Evaluation metrics calculation
├── utils.py              # Utility functions for dataset loading, text cleaning, etc.
├── data/                 # Directory containing datasets
├── data_util/            # Data preprocessing utilities
├── ATTACKS/              # Security attack simulation scripts
├── lcc/                  # Local cache coordination modules
├── lcc_cache/            # Cache management components
├── PPVFD/                # Privacy-preserving vertical federated learning modules
├── PPVFedCache/          # Federated caching implementation for PPVFD
├── resnet_client/        # Client-side ResNet model implementation
├── _pycache__/           # Python compiled cache (auto-generated)
└── idea/                 # IDE-specific configuration files
```

## Usage
### Navigate to the project directory
**Switch to the project root directory**
```
cd peiy/paper
```
###  Run the Main Program
**Execute Federated Cache Main Script**
```
python3 main_fedcache.py
```
###  Automated Script Execution
```
#!/bin/bash  
set -e  # 遇到错误自动退出  
  
# 创建 Conda 环境（如果不存在）  
if ! conda env list | grep -q "peiy"; then  
    conda create -n peiy python=3.8 -y   
    
# 激活环境  
source $(conda info --base)/etc/profile.d/conda.sh  
conda activate peiy  
  
# 安装依赖（带错误检查）  
pip install torch torchvision matplotlib scikit-learn hnswlib wandb  
  
#  运行程序（带路径检查）  
cd peiy/paper || { echo "Error: peiy/paper directory missing!"; exit 1; }  
python3 main_fedcache.py
```
**Execute Federated Cache Main Script**

```
chmod +x run.sh  
```
**Run the Script**

``` 
./run.sh
```

