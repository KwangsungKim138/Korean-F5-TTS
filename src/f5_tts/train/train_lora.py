# LoRA fine-tuning: load pretrained base, apply LoRA (r=64, alpha=16), train only LoRA params.
# Usage: accelerate launch src/f5_tts/train/train_lora.py --config-name F5TTS_Base_ft datasets.name=KSS_full
# Ensure save_dir contains pretrained_*.pt (or set ckpts.pretrained_path).

import gc
import os
from importlib.resources import files

import hydra
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, TaskType

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


# LoRA defaults (CoreaSpeech-style)
LORA_R = 64
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]  # DiT attention linears


def _load_pretrained_into_model(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load base weights from checkpoint (ema_model_state_dict) into model. Strips 'ema_model.' prefix."""
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = dict(load_file(ckpt_path, device="cpu"))
        ckpt = {"ema_model_state_dict": state}
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "ema_model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint must contain 'ema_model_state_dict': {ckpt_path}")
    state = ckpt["ema_model_state_dict"]
    state_dict = {
        k.replace("ema_model.", ""): v
        for k, v in state.items()
        if k not in ("initted", "update", "step")
    }
    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        state_dict.pop(key, None)
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state, state_dict
    gc.collect()


def _find_pretrained_ckpt(checkpoint_path: str) -> str | None:
    """Return path to pretrained_*.pt in checkpoint_path, or None."""
    if not os.path.isdir(checkpoint_path):
        return None
    for f in os.listdir(checkpoint_path):
        if f.startswith("pretrained_") and f.endswith((".pt", ".safetensors")):
            return os.path.join(checkpoint_path, f)
    return None


os.chdir(str(files("f5_tts").joinpath("../..")))


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}_lora"
    wandb_resume_id = model_cfg.ckpts.get("wandb_resume_id", None)
    ds_name = model_cfg.datasets.name
    data_scale = (
        ds_name.replace("KSS_", "")
        if isinstance(ds_name, str) and ds_name.startswith("KSS_")
        else "full"
    )

    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    model = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        ),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    checkpoint_path = str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}"))
    pretrained_path = model_cfg.ckpts.get("pretrained_path") or _find_pretrained_ckpt(checkpoint_path)
    if not pretrained_path or not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"LoRA needs a pretrained checkpoint. Put pretrained_*.pt in {checkpoint_path} or set ckpts.pretrained_path."
        )
    print(f"Loading pretrained weights from {pretrained_path}")
    _load_pretrained_into_model(model, pretrained_path)

    lora_r = model_cfg.ckpts.get("lora_r", LORA_R)
    lora_alpha = model_cfg.ckpts.get("lora_alpha", LORA_ALPHA)
    lora_targets = model_cfg.ckpts.get("lora_target_modules", LORA_TARGET_MODULES)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=model_cfg.ckpts.get("wandb_project", "F5-TTS"),
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict={
            **OmegaConf.to_container(model_cfg, resolve=True),
            "data_scale": data_scale,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        },
        resume_from_checkpoint=False,
    )

    train_dataset = load_dataset(
        model_cfg.datasets.name,
        tokenizer,
        mel_spec_kwargs=model_cfg.model.mel_spec,
    )
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
