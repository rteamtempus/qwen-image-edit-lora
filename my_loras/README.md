# Drop your LoRA files here

Put your `.safetensors` LoRA files in this folder. At build time they're copied
into ComfyUI's `models/loras/` (see the Dockerfile), so you can use them by
filename in a request, e.g.:

```json
"loras": [{ "name": "my_style.safetensors", "strength": 0.8 }]
```

Notes:
- **Each file must be under 100 MB** (GitHub's per-file limit). For larger LoRAs,
  host them on Hugging Face and use the `wget` example in the Dockerfile instead.
- After adding/removing files here, commit + push, then redeploy the endpoint.
