# Adding LoRAs to this Qwen Image Edit worker

This fork adds **LoRA support** to the worker. You can apply one or more LoRAs on
top of the model when editing an image.

## What changed

- **`handler.py`** now accepts LoRAs in the request and chains them into the
  ComfyUI graph (right before `ModelSamplingAuraFlow`, stacking on top of the
  built-in Lightning speed LoRA). No effect when you don't pass any.
- **`entrypoint.sh`** auto-detects a RunPod **Network Volume** and lets ComfyUI
  load LoRAs from `<volume>/loras` — so you can add LoRAs **without rebuilding**.
- **`Dockerfile`** has a marked spot to bake LoRAs into the image via `wget`.

## Request format

Reference LoRAs by **filename only**. Either form works:

```jsonc
// one or more
"loras": [
  { "name": "my_style.safetensors", "strength": 0.9 },
  { "name": "another.safetensors",  "strength": 0.6 }
]

// or the single-LoRA shorthand
"lora": "my_style.safetensors",
"lora_strength": 0.9
```

Full example:

```json
{
  "input": {
    "prompt": "make it an oil painting",
    "image_base64": "<...>",
    "loras": [{ "name": "oil_painting_style.safetensors", "strength": 0.8 }]
  }
}
```

`strength` defaults to `1.0`. Tip: the worker runs in 4 steps (Lightning LoRA),
so very strong style LoRAs can fight it — start around `0.6–0.9`.

## Getting your LoRA files onto the endpoint

### Option A — bake into the image (simplest first run)
Best when your LoRA is at a **direct download URL**.
1. In the **`Dockerfile`**, find the `YOUR LoRAs` block and uncomment a line:
   ```dockerfile
   RUN wget -q "https://huggingface.co/USER/REPO/resolve/main/my_style.safetensors" \
        -O /ComfyUI/models/loras/my_style.safetensors
   ```
   (Hugging Face `resolve` links work as-is. Civitai needs `?token=YOUR_API_TOKEN`.)
2. Rebuild/redeploy (below). Reference it as `my_style.safetensors`.

### Option B — Network Volume (no rebuild to add more later)
1. In RunPod: **Storage → Network Volume** (same region as your endpoint), attach
   it to the endpoint.
2. Put your `.safetensors` files in the volume's **`loras/`** folder.
3. Reference them by filename. `entrypoint.sh` already points ComfyUI at the volume.

## Deploying (no Docker needed — RunPod builds from GitHub)

1. **Put this code on your own GitHub.** From this folder:
   ```bash
   git remote remove origin
   git remote add origin https://github.com/<YOU>/qwen-image-edit-lora.git
   git add -A
   git commit -m "Add LoRA support"
   git branch -M main
   git push -u origin main
   ```
   (Create the empty repo on github.com first.)
2. In **RunPod → Serverless → New Endpoint → “Import Git Repository”**, connect
   GitHub and pick your repo/branch. RunPod builds the image for you.
3. (If using a volume) attach your Network Volume to the endpoint.
4. Copy the new **Endpoint ID** into the app's `.env` as `RUNPOD_ENDPOINT_ID`
   and restart the app.

That's it — the app's edit requests can then include `loras`.
