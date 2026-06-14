# Use specific version of nvidia cuda image
FROM wlsdml1114/multitalk-base:1.7 as runtime

# wget 설치 (URL 다운로드를 위해)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa

# --- VLM planner stage (per-subject randomized edits) ------------------------
# The planned pipeline (handler.run_planned_edit → planner.py) loads a Qwen3-VL
# instruct model via transformers to locate each person and write an individually
# grounded directive. accelerate is needed for device_map; transformers must be
# new enough to support Qwen3-VL.
#
# ⚠️ VERIFY: upgrading transformers can affect ComfyUI custom nodes. If a node
# breaks, pin transformers to a version that supports BOTH (test the image), or
# move the planner to its own endpoint. The base image already ships a
# transformers; we upgrade it for Qwen3-VL support.
RUN pip install -U "accelerate>=0.34" "transformers>=4.57.0"
#
# Bake the VLM weights INTO the image (same pattern as the base diffusion models
# below) so the worker needs no Network Volume and isn't pinned to a region.
# Downloaded once at build time via hf_transfer; PLANNER_MODEL_ID points planner.py
# at the local copy so there's zero HF call (and zero re-download) at runtime.
# Size lever: swap to Qwen/Qwen3-VL-4B-Instruct (~half the size) if the image gets
# too big or builds time out — it's usually enough for grounding + directives.
# Pull the VLM at RUNTIME (on the first planned request), NOT at build time.
# Baking a 16GB model blew the 30-minute hub build limit AND triggered flaky
# registry layer-commit errors. Downloading at runtime keeps the image small so
# builds are fast and reliable, and lets you swap the planner model by changing
# PLANNER_MODEL_ID on the endpoint with NO rebuild.
#
# Default is the 4B (~8GB, plain bf16 — loads with no FP8 kernels) to keep the
# one-time per-worker download light. To use a bigger/different VLM, just set
# PLANNER_MODEL_ID on the endpoint (e.g. Qwen/Qwen3-VL-8B-Instruct). With no
# Network Volume attached it caches on the worker's local disk (region-free);
# use min-workers=1 during testing so the download only happens once.
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PLANNER_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct

# Set working directory
WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install --no-cache-dir -r requirements.txt

# Download base models in PARALLEL (total time ≈ the largest file, not the sum of
# all four) so the build stays well under RunPod's build time limit. `set -e` +
# `wait $pN` makes the build fail if any single download fails.
RUN set -e; \
    wget -q https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors -O /ComfyUI/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors & p1=$!; \
    wget -q https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors -O /ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors & p2=$!; \
    wget -q https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors -O /ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors & p3=$!; \
    wget -q https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors -O /ComfyUI/models/vae/qwen_image_vae.safetensors & p4=$!; \
    wait $p1; wait $p2; wait $p3; wait $p4

# ============================================================================
# YOUR LoRAs (baked into the image)
# ----------------------------------------------------------------------------
# Small LoRAs (<100MB) can be dropped into ./my_loras and committed.
COPY my_loras/ /ComfyUI/models/loras/
#
# Your LoRAs, pulled from Hugging Face at build time. Downloaded in PARALLEL (the
# trailing & runs them at once; `wait` fails the build if any download fails) so
# the build stays under the time limit. To add/remove a baked LoRA, edit these
# lines AND the KNOWN_LORAS chips in the app. Reference each by filename.
RUN set -e; \
    wget -q "https://huggingface.co/Rt5556/qwen-loras/resolve/main/lora_1.safetensors"           -O /ComfyUI/models/loras/lora_1.safetensors           & p1=$!; \
    wget -q "https://huggingface.co/Rt5556/qwen-loras/resolve/main/BNElora_2.safetensors"        -O /ComfyUI/models/loras/BNElora_2.safetensors        & p2=$!; \
    wget -q "https://huggingface.co/Rt5556/qwen-loras/resolve/main/qwen_MCNL_v1.0.safetensors"   -O /ComfyUI/models/loras/qwen_MCNL_v1.0.safetensors   & p3=$!; \
    wait $p1; wait $p2; wait $p3
# ============================================================================

COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]