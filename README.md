## SLAM Cell Segmentation

模型训练结果在`outputs`
推荐使用 `outputs/cell_balanced_last/top5_epoch0129_valloss18.550179.pt`

运行环境：与 [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3) 相同
同时需要预先下载sam3的`sam3/sam3.pt`和`sam3/assets/bpe_simple_vocab_16e6.txt.gz` 

### 推理方法

首先用`preprocess.py`对于四通道的原tif进行预处理，图像会输出到一个文件夹，其中是每帧的预处理后rgb格式的图片
之后使用`infer_folder_overlay.py`进行推理

### infer_folder_overlay.py 使用说明

推理代码在`infer_folder_overlay.py`
其中比较关键的参数在第521行 
```python
    processor_main = Sam3Processor(model_main, confidence_threshold=0.4)
```
`confidence_threshold`控制了对于最终分割结果的选择，经验值为0.3-0.5之间合适。如果感觉分割到的细胞太少可以调低，反之，如果感觉分割到的细胞过多或者将许多并不是细胞的也分割进去，可以调高。

该脚本会：
1. 读取一个输入 TIF 作为底图帧序列
2. 对 `img_dir` 文件夹里的图片逐帧做推理
3. 将推理得到的实例 mask 叠加到对应的 TIF 帧上，并保存：
   - 原始底图帧（PNG）
   - 叠加可视化结果（PNG）
   - （可选）每帧实例 id 对应 score 的 jsonl（代码里目前注释了写入）


#### 输入数据要求

##### `--in_tif`（底图 TIF）

最后overlay显示的底图
支持的 TIF shape：

- `(T, 3, H, W)`
- `(T, H, W, 3)`
- `(3, H, W)`
- `(H, W, 3)`

> 注意：最终要求通道数 C=3（RGB）。

##### `--img_dir`（推理图片文件夹）

应当`preprocess.py`的结果文件夹路径
文件夹内图片按文件名排序逐帧匹配 TIF：
- 支持后缀：`.jpg .jpeg .png .tif .tiff .bmp`
- 推理帧数 `N = min(len(images), T)`

#### 推理命令行

```bash
python infer_folder_overlay.py \
  --in_tif /path/to/base.tif \
  --img_dir /path/to/infer_images \
  --out_dir /path/to/output \
  --prompt cell \
  --ckpt /path/to/sam3_checkpoint.pt \
  --bpe /path/to/bpe.model \
  --lora_config /path/to/lora_config.yaml \
  --lora_weight /path/to/lora_weights.pt

```

运行与输出举例：

```bash
python infer_folder_overlay.py  --in_tif real_rgb/2024-10-23_161048_leg2-2.tif --img_dir slam_seg/output/composite_rgb_try/2024-10-23_161048_leg2_rgb_frames   --out_dir inference/2024-10-23_161048_leg2_1008   --prompt "cell"   --ckpt /home/student002/.cache/modelscope/hub/models/facebook/sam3/sam3.pt   --bpe /home/student002/MIT_segmetation/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz   --lora_config configs/cell.yaml   --lora_weight outputs/cell_balanced_last/top5_epoch0117_valloss18.505558.pt
Replaced 31 nn.MultiheadAttention modules with MultiheadAttentionLoRA
Applied LoRA to 124 modules:
  - transformer.encoder.layers.0.self_attn.q_proj
  - transformer.encoder.layers.0.self_attn.k_proj
  - transformer.encoder.layers.0.self_attn.v_proj
  - transformer.encoder.layers.0.self_attn.out_proj
  - transformer.encoder.layers.0.cross_attn_image.q_proj
  - transformer.encoder.layers.0.cross_attn_image.k_proj
  - transformer.encoder.layers.0.cross_attn_image.v_proj
  - transformer.encoder.layers.0.cross_attn_image.out_proj
  - transformer.encoder.layers.1.self_attn.q_proj
  - transformer.encoder.layers.1.self_attn.k_proj
  - transformer.encoder.layers.1.self_attn.v_proj
  - transformer.encoder.layers.1.self_attn.out_proj
  - transformer.encoder.layers.1.cross_attn_image.q_proj
  - transformer.encoder.layers.1.cross_attn_image.k_proj
  - transformer.encoder.layers.1.cross_attn_image.v_proj
  ... and 109 more
Loaded LoRA weights from /home/student002/MIT_segmetation/SAM3_LoRA/outputs/cell_balanced_last/top5_epoch0117_valloss18.505558.pt
[INFO] t=10/53 kept_instances=65 added_sam3=0
[INFO] t=20/53 kept_instances=62 added_sam3=0
[INFO] t=30/53 kept_instances=71 added_sam3=0
[INFO] t=40/53 kept_instances=67 added_sam3=0
[INFO] t=50/53 kept_instances=46 added_sam3=0
[INFO] t=53/53 kept_instances=43 added_sam3=0
[OK] originals -> inference/2024-10-23_161048_leg2_1008/originals_png
[OK] overlays   -> inference/2024-10-23_161048_leg2_1008/overlays_png
[OK] scores     -> inference/2024-10-23_161048_leg2_1008/scores.jsonl

```