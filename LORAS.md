# LoRAs & custom models on this Qwen Image Edit worker

This fork lets you apply LoRAs (and swap the base model) at **request time**,
loading the files from an attached **RunPod Network Volume** — so you mix-and-match
freely with **no image rebuilds**.

## What changed
- **`handler.py`** accepts `loras` (and `model_name`/`model_url`) in the request and
  wires them into the ComfyUI graph. LoRAs stack on top of the built-in Lightning
  speed LoRA. References can be a **filename** (already on the volume) or a **URL**
  (downloaded once and cached onto the volume).
- **`entrypoint.sh`** auto-detects the Network Volume and points ComfyUI at
  `<volume>/loras`, `<volume>/diffusion_models`, etc.

## Request format

```jsonc
{
  "input": {
    "prompt": "make it an oil painting",
    "image_base64": "<...>",

    // LoRAs — by filename (staged on the volume) and/or by URL (cached on first use)
    "loras": [
      { "name": "oil_style.safetensors", "strength": 0.8 },
      { "url": "https://huggingface.co/USER/REPO/resolve/main/grain.safetensors", "strength": 0.5 }
    ],

    // optional: swap the base model (must be Qwen-Image-Edit compatible)
    "model_name": "my_qwen_finetune.safetensors"
  }
}
```

Shorthand for a single LoRA: `"lora": "oil_style.safetensors"`, `"lora_strength": 0.8`.
`strength` defaults to `1.0`. The worker runs in 4 steps (Lightning LoRA), so
strong style LoRAs often look best around **0.6–0.9**.

## Putting files on the Network Volume

Your files live in folders on the volume (mounted at `/runpod-volume` in the worker):
- LoRAs → `loras/`
- Custom base models → `diffusion_models/`

Two ways to get them there:

1. **Pass a URL once** — easiest. Send a `url` (LoRA) or `model_url`. The worker
   downloads it into the volume and **caches it**; reference it by filename after.
2. **Upload directly** — spin up a cheap temporary Pod that mounts the volume,
   drop your `.safetensors` into `loras/` (or `diffusion_models/`) via its file
   browser/terminal, then stop the Pod.

After a file is on the volume, every future request/worker can use it by filename
with **no download and no rebuild**.

## Deploying (RunPod builds from GitHub)
1. Push this repo to your GitHub (already done if you're reading this in your fork).
2. **Serverless → New Endpoint → Import Git Repository →** your repo. Set the region
   to where your **Network Volume** lives (e.g. US-NC-1) and **attach the volume**.
3. Copy the **Endpoint ID** into the app's `.env` (`RUNPOD_ENDPOINT_ID`) and restart.

> Custom models must be **Qwen-Image-Edit-compatible** (same architecture), or the
> text-encoder/VAE nodes won't match. Full models are large (~20 GB) — keep them on
> the volume; don't expect to download one per request.
