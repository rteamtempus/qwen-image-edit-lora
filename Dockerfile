# Use specific version of nvidia cuda image
FROM wlsdml1114/multitalk-base:1.7 as runtime

# wget 설치 (URL 다운로드를 위해)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa

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

# Download models
RUN wget -q https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors -O /ComfyUI/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors
RUN wget -q https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors -O /ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
RUN wget -q https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors -O /ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors 
RUN wget -q https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors -O /ComfyUI/models/vae/qwen_image_vae.safetensors

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
    wget -q "https://huggingface.co/Rt5556/qwen-loras/resolve/main/lora_1.safetensors"      -O /ComfyUI/models/loras/lora_1.safetensors      & p1=$!; \
    wget -q "https://huggingface.co/Rt5556/qwen-loras/resolve/main/FElora_3.safetensors"     -O /ComfyUI/models/loras/FElora_3.safetensors     & p2=$!; \
    wget -q "https://huggingface.co/Rt5556/qwen-loras/resolve/main/natural_skin.safetensors" -O /ComfyUI/models/loras/natural_skin.safetensors & p3=$!; \
    wait $p1; wait $p2; wait $p3
# ============================================================================

COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]