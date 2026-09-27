# Wan2.2 Prompt Relay：重叠时间段

这次改动接入原来的 **T2V-A14B 文生视频**流程，高噪声和低噪声两个模型都使用同一份时间安排，不需要重新训练。

## 最直观的例子

可以同时填写：

- 第一句：“劫匪走出银行，跑到汽车旁，打开车门进入车内。”时间是 `[0, 3)` 秒。
- 第二句：“劫匪和汽车后方的银行发生爆炸，喷出烟雾和碎片。”时间是 `[2, 4)` 秒。

这样，2 到 3 秒时，两句话都能参与引导画面。它们可以描述不同物体。
程序负责控制时间，不负责把文字绑定到具体像素；主体、前后位置仍需在提示词中写清楚，也不能保证模型一定生成成功。

示例文件：[prompt_relay_overlap.json](prompt_relay_overlap.json)。在 Wan2.2 目录运行：

```bash
python generate.py --task t2v-A14B --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --size "832*480" --frame_num 81 --offload_model True --convert_model_dtype \
  --prompt_filepath prompt_relay_overlap.json
```

将模型路径换成自己的路径，并按原项目要求安装依赖和准备权重。

## 如何填写时间

```json
{
  "global_prompt": "银行外的连续广角镜头，汽车在前景，银行在后方。",
  "local_prompts": ["劫匪走出银行并进入汽车。", "后方的银行发生爆炸。"],
  "segment_intervals": [[0, 3], [2, 4]],
  "time_unit": "seconds"
}
```

第一组时间对应第一句话，第二组对应第二句话。支持部分重叠、完全重叠、嵌套、中间留空，时间也不必按开始顺序排列。结束点不包含在区间内。

`time_unit` 可选 `seconds`（秒）或 `internal_frame`（模型内部时间格，默认）。内部时间格必须填写整数；不要同时填写旧的 `segment_lengths`。

秒数按照视频实际导出的 `config.sample_fps` 换算。T2V-A14B 默认是 16 FPS，每 4 帧对应一个内部时间格，即约 0.25 秒。上例会转换为内部 `[0, 12)` 和 `[8, 16)`，程序会打印换算结果。不要在 JSON 里另填 `fps`。时间格只是安排提示词的尺度，生成画面边界未必精确到某一帧。

可选 `tail_width` 控制提示词在区间外多远衰减到 `epsilon`，单位与时间段相同。默认宽度为两个内部时间格，`epsilon` 默认是 `0.001`。区间不能超出视频时长；短到没有覆盖任何内部时间格时会报错，提示你加宽。

## 不填时间会怎样

**按你确认的要求，不会根据文字猜测重叠，也不会默认安排重叠。**

不填 `segment_intervals` 时，继续使用原来的顺序安排：有 `segment_lengths` 就按长度累计；没有就按原来的向上取整步长分配，末段截到视频结尾。某句话分不到有效时间格时会报错。旧参数的 attention 衰减设置保留。

Wan 这个分支仅接入手动指定重叠时间；没有移植 Hunyuan 的可选 `auto_overlap` 分配规则。想要重叠，请填写 `segment_intervals`。

## 程序具体改了什么

1. 每句话保存自己的开始和结束时间，不再只能把上一句话的结束作为下一句话的开始。
2. 重叠区域同时放行多句话对应的 attention。区间外仍按距离逐渐减弱。
3. 同一句话重复出现时，分别找到各自的文字位置，避免控制到错误的提示词。
4. 多卡分段计算时，使用整段视频的时间位置，避免每张卡都从第 0 秒重新算。
5. 对超出时长、无效区间、文字被 T5 长度上限截断等情况给出明确错误。JSON 支持 UTF-8。

Attention 的基本公式仍然是 `softmax(QKᵀ / √d − 时间惩罚) V`。变化在于时间惩罚可以同时容纳多段有效时间。多句话仍共享同一个 softmax，权重不一定相同，也不是给各句话单独生成一个物体。

该入口用于 `t2v-A14B`，暂未接入其他 Wan 任务。使用提示词 JSON 时请关闭 `--use_prompt_extend`。不传 `--prompt_filepath` 则不启用 Prompt Relay；视频自注意力的滑动窗口开关独立控制。

## 已有验证与边界

### 与 sliding window 一起使用

在 `feat/prompt-relay-sliding-window` 分支中，T2V-A14B 还可以开启视频自注意力滑动窗口：

```bash
python generate.py --task t2v-A14B --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --size "832*480" --frame_num 81 --offload_model True --convert_model_dtype \
  --prompt_filepath prompt_relay_overlap.json \
  --sliding_window --window_length 12 --window_stride 6
```

Prompt Relay 管“每个时间段看哪句话”，滑动窗口管“每帧能看附近哪些视频帧”。
两者可以同时使用，提示词时间不会在每个窗口重新从零开始。窗口重叠处的输出取平均。

三个参数分别是开关、窗口长度、滑动步长。长度和步长按**内部 latent 帧**计数；
默认是关闭、31、16。`--sliding_window false` 可以显式关闭。
81 个输出帧对应 21 个内部帧，因此默认长度 31 会覆盖整段视频；上面的 12/6 才会实际分窗。
更长视频可以使用 `--frame_num 241`，其内部长度为 61 帧，并根据完整视频时长设置提示词区间。
要求 `0 < window_stride <= window_length`，避免出现无人覆盖的帧。

滑动窗口不等于无限延长视频：完整 latent、模型权重和 VAE 解码仍占显存，
较远时间点之间的直接联系也会减弱。完整出片效果、显存和速度需要用实际模型权重对比。
详见 [滑动窗口说明与验证步骤](SLIDING_WINDOW.md)。

### Overlapping 回归测试

```bash
python -m unittest discover -s tests -p test_prompt_relay.py -v
```

覆盖时间换算、重叠区域、重复文字、文本截断、旧版规则，以及 CPU/CUDA 多种精度下的 attention 数值比较。另用小型真实 Wan 网络检查接入，用模拟分卡检查时间偏移。

小型网络测试用普通 attention 替代 FlashAttention 的自注意力，分布式通信使用模拟。这些检查没有运行完整预训练模型出片，也没有运行真实多卡推理；最终视频效果仍需用模型权重实测。
