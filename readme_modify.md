# 代码修改说明
代码主要修改了dmpfold/network.py

# 关于ConvNeXtV2_Block调用
修改dmpfold/network.py 第 353 行代码将use_convnext_v2_block设置为 True，默认为 False 默认只用一个 conv 设置为 True 则引入convnext_v2_block 可设置convnext_drop_path_ratio，以及修改第 343 行代码

# 关于 self-attention 调用
修改dmpfold/network.py  第 353 行代码将use_self_attention设置为True，默认为 False默认会调用 SCSE 模块 self-attention 的参数可在第 238 行修改主要修改 heads dim_head dropout 如果heads dim_head太大容易显存溢出