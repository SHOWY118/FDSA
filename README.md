#  联邦蒸馏中的隐蔽性操纵漏洞探究
这是论文《联邦蒸馏中的隐蔽性操纵漏洞探究》的官方代码仓库
  
## 概述
  
-------------------------  
  
-   **核心发现**：FDSA通过精妙操纵客户端logits（对数概率输出），既能规避检测，又能显著降低服务端聚合过程的鲁棒性。
  
![image-20250526222704042](/FDSA/Overflow_of_FDSA.png)

## 依赖环境
* Python 3.8  
* PyTorch (`torch`)  2.7.0  
* PyTorch (`torchvision`) 0.22.0
* wandb 0.19.11
* hnswlib 0.8.0
* matplotlib 3.10.3
* Scikit-learn (`scikit-learn`)  1.6.1
* NumPy (`numpy`)  2.1.3

##  安装与配置
### 1.  创建并激活Conda环境
**创建名为'peiy'的Conda环境（确保Python版本与项目兼容）**

```
conda create -n peiy python=3.8 -y
```
**激活环境**
```
conda activate peiy
```

### 2.  安装依赖项
**安装PyTorch核心依赖**
```
pip install torch torchvision
```
**安装其他必要依赖**
```
pip install matplotlib          # 绘图库 
pip install scikit-learn        # 机器学习工具
pip install hnswlib             # 高效向量搜索库
pip install wandb               # 实验跟踪工具
```
## 文件结构
```  
RecServe/  
├── main.py # Main script for running experiments  
├── recursive_serve.py # Core implementation of the RecServe framework  
├── base_serve.py # Basic serving class  
├── evaluation.py # Evaluation metrics calculation  
├── utils.py # Utility functions for dataset loading, text cleaning, etc.  
└── README.md # This document  
```

##  使用指南
### 进入项目目录
**切换至项目根目录**
```
cd peiy/paper
```
### 运行主程序
**执行联邦缓存主脚本**
```
python3 main_fedcache.py
```
### 自动化脚本执行
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
**赋予执行权限**

```
chmod +x run.sh  
```
**运行自动化脚本**

``` 
./run.sh
```
