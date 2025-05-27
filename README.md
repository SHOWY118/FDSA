# Exploring Subtle Manipulation Vulnerabilities in Federated Distillation
### Overview  
  
-------------------------  
  
- **Take-Away**: FDSA subtly manipulates client logits to evade detection while significantly degrading the robustness of server-side aggregation.  
  
![image-20250526222704042](/FDSA/Overflow_of_FDSA.png)

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
**Create a Conda environment named 'xx' (ensure Python version is compatible with the project).**

```
conda create -n xx python=3.8 -y
```
**Activate the environment**
```
conda activate xx
```

### 2. Install dependencies
 **Install PyTorch Dependencies**
```
pip install torch torchvision
```
**Install Additional Dependencies**
```
pip install matplotlib          # Plotting library  
pip install scikit-learn        # Machine learning toolkit  
pip install hnswlib             # Efficient vector search library  
pip install wandb               # Experiment tracking tool  
```
## File Structure
```
FDSA/  
├── main_fdsa.py # Main script for running experiments  
├── PPVFD.py # Core implementation for FD  
├── PPVFedCache.py # Core implementation for FedCache  
├── ATTACKS.py # Attacks methods  
├── lcc.py # Tools for FD.  
├── lcc_cache.py # Tools for FedCache.  
└── README.md # This document
```

## Usage
### Navigate to the project directory
**Switch to the project root directory**
```
cd xx/paper
```
###  Run the Main Program
**Execute Federated Cache Main Script**
```
python3 main_fedcache.py
```
###  Automated Script Execution
```
#!/bin/bash  
set -e  
  
#  Create a Conda environment named 'my_env' with Python 3.9 (if it doesn't exist) 
if ! conda env list | grep -q "xx"; then  
    conda create -n xx python=3.8 -y   
    
#  Activate the environment 
source $(conda info --base)/etc/profile.d/conda.sh  
conda activate xx  
  
# Install dependencies (with error checking)
pip install torch torchvision matplotlib scikit-learn hnswlib wandb  
  
# Run program (with path checking)
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

