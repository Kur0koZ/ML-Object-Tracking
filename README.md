# Requirement

Dataset.zip (Roboflow)

# Download
python -m pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics==8.4.163

# Check If Use GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
