# FDSA
#How to Run the Code


###1. 创建并激活 Conda 环境
# 创建名为 'peiy' 的 Conda 环境（Python 版本需与项目兼容）
conda create -n peiy python=3.8 -y

# 激活环境
conda activate peiy


###2. 安装依赖库
# 安装 PyTorch 相关依赖（根据 CUDA 版本调整，此处为默认 CPU 版本）
pip install torch torchvision

# 安装其他依赖
pip install matplotlib          # 绘图库
pip install scikit-learn        # 机器学习工具
pip install hnswlib             # 高效向量搜索库
pip install wandb               # 实验跟踪工具

###3. 进入项目目录
# 切换到项目主目录
cd peiy/paper


###4. 运行主程序
# 执行联邦缓存主脚本
python3 main_fedcache.py


###自动化脚本：
#!/bin/bash
set -e  # 遇到错误自动退出

# 创建 Conda 环境（如果不存在）
if ! conda env list | grep -q "peiy"; then
    conda create -n peiy python=3.8 -y
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate peiy

# 安装依赖（带错误检查）
pip install torch torchvision matplotlib scikit-learn hnswlib wandb

#  运行程序（带路径检查）
cd peiy/paper || { echo "Error: peiy/paper directory missing!"; exit 1; }
python3 main_fedcache.py

# 赋予脚本执行权限
chmod +x run.sh
# 执行脚本
./run.sh
