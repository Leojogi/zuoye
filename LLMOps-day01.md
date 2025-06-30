**作业**
```
# 初始化隔离环境
conda create -p /root/autodl-tmp/sglang python=3.11
conda activate /root/autodl-tmp/sglang

# 安装sglang
pip install sglang[all]

# 使用sglang来下载并启动大模型
SGLANG_USE_MODELSCOPE=true python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --trust-remote-code --tp 1 --api-key magedu.com --served-model-name DeepSeek-R1-8B

# 安装open-weiui
conda create -p /root/autodl-tmp/open-webui python=3.11
conda activate /root/autodl-tmp/open-webui
pip install open-webui

# 启动open-webui
open-webui serve

# 隧道到云主机
ssh -CNg -L 8080:127.0.0.1:8080 root@connect.nmb2.seetacloud.com -p 11764
//ssh -p 11764 root@connect.nmb2.seetacloud.com
```
