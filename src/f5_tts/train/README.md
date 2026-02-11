# Training

Check your FFmpeg installation:
```bash
ffmpeg -version
```
If not found, install it first (or skip assuming you know of other backends available).

## Prepare Dataset

Example data processing scripts, and you may tailor your own one along with a Dataset class in `src/f5_tts/model/dataset.py`.

### 1. Some specific Datasets preparing scripts
Download corresponding dataset first, and fill in the path in scripts.

```bash
# Prepare the Emilia dataset
python src/f5_tts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py

# Prepare the LibriTTS dataset
python src/f5_tts/train/datasets/prepare_libritts.py

# Prepare the LJSpeech dataset
python src/f5_tts/train/datasets/prepare_ljspeech.py
```

### 2. Create custom dataset with CSV
Prepare a CSV with two columns using a required header: `audio_file|text`. Audio paths must be absolute.
Use guidance see [#57 here](https://github.com/SWivid/F5-TTS/discussions/57#discussioncomment-10959029).

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /path/to/output
```

## Training & Finetuning

Once your datasets are prepared, you can start the training process.

### 1. Training script used for pretrained model

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config

# .yaml files are under src/f5_tts/configs directory
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml

# possible to overwrite accelerate and hydra config
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++datasets.batch_size_per_gpu=19200
```

### 2. Finetuning practice
Discussion board for Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI training/finetuning with `src/f5_tts/train/finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

If want to finetune with a variant version e.g. *F5TTS_v1_Base_no_zero_init*, manually download pretrained checkpoint from model weight repository and fill in the path correspondingly on web interface.

If use tensorboard as logger, install it first with `pip install tensorboard`.

<ins>The `use_ema = True` might be harmful for early-stage finetuned checkpoints</ins> (which goes just few updates, thus ema weights still dominated by pretrained ones), try turn it off with finetune gradio option or `load_model(..., use_ema=False)`, see if offer better results.

### 3. LoRA fine-tuning (pretrained + N2gk+ → g2p → allophone)

Config: `src/f5_tts/configs/F5TTS_Base_ft_Lora.yaml`. Pretrained ckpt and vocab paths are set there; data path is `datasets.load_path`.

**1) 사전학습 모델·vocab 위치**

- `ckpts/pretrained/model_pretrained_1200000.pt` — 사전학습 모델
- `ckpts/pretrained/vocab_pretr.txt` — vocab (config에 이미 설정됨)

**2) 데이터셋: N2gk+ → g2p → allophone**

프로젝트 루트에서:

```bash
# KSS transcript 기준 (기본 출력: data/KSS_n2gk_allophone)
python src/f5_tts/train/datasets/prepare_kss_n2gk_allophone.py --name KSS_n2gk_allophone --transcript data/KSS/transcript.v.1.4.txt

# Parquet 입력이면
python src/f5_tts/train/datasets/prepare_kss_n2gk_allophone.py --name KSS_n2gk_allophone --parquet /path/to/data.parquet --audio-base /path/to/wavs --write-parquet
```

생성되는 디렉터리: `data/KSS_n2gk_allophone/` (raw.arrow, duration.json, vocab.txt).

**3) LoRA 학습 실행**

```bash
accelerate launch src/f5_tts/train/train_lora.py --config-name F5TTS_Base_ft_Lora
```

config에 이미 다음이 들어 있음:

- `ckpts.pretrained_path: ckpts/pretrained/model_pretrained_1200000.pt`
- `model.tokenizer_path: ckpts/pretrained/vocab_pretr.txt`
- `datasets.load_path: data/KSS_n2gk_allophone`

경로만 바꿀 경우 예:

```bash
accelerate launch src/f5_tts/train/train_lora.py --config-name F5TTS_Base_ft_Lora ckpts.pretrained_path=/path/to/model.pt model.tokenizer_path=/path/to/vocab.txt datasets.load_path=data/MyN2gkAllophone
```

체크포인트 저장 위치: `ckpts/F5TTS_Base_vocos_custom_KSS_n2gk_allophone_lora/`.

### 4. W&B Logging

The `wandb/` dir will be created under path you run training/finetuning scripts.

By default, the training script does NOT use logging (assuming you didn't manually log in using `wandb login`).

To turn on wandb logging, you can either:

1. Manually login with `wandb login`: Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)
2. Automatically login programmatically by setting an environment variable: Get an API KEY at https://wandb.ai/authorize and set the environment variable as follows:

On Mac & Linux:

```
export WANDB_API_KEY=<YOUR WANDB API KEY>
```

On Windows:

```
set WANDB_API_KEY=<YOUR WANDB API KEY>
```
Moreover, if you couldn't access W&B and want to log metrics offline, you can set the environment variable as follows:

```
export WANDB_MODE=offline
```
