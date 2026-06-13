import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import urllib.error
import binascii # Base64 에러 처리를 위해 import
import subprocess
import time
import hashlib


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA 검사 및 설정
def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# CUDA 검사 실행
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)



server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
def save_data_if_base64(data_input, temp_dir, output_filename):
    """
    입력 데이터가 Base64 문자열인지 확인하고, 맞다면 파일로 저장 후 경로를 반환합니다.
    만약 일반 경로 문자열이라면 그대로 반환합니다.
    """
    # 입력값이 문자열이 아니면 그대로 반환
    if not isinstance(data_input, str):
        return data_input

    try:
        # Base64 문자열은 디코딩을 시도하면 성공합니다.
        decoded_data = base64.b64decode(data_input)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # 디코딩에 성공하면, 임시 파일로 저장합니다.
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f: # 바이너리 쓰기 모드('wb')로 저장
            f.write(decoded_data)
        
        # 저장된 파일의 경로를 반환합니다.
        print(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path

    except (binascii.Error, ValueError):
        # 디코딩에 실패하면, 일반 경로로 간주하고 원래 값을 그대로 반환합니다.
        print(f"➡️ '{data_input}'은(는) 파일 경로로 처리합니다.")
        return data_input
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        # ComfyUI returns the validation detail (bad node/value, missing LoRA, etc.)
        # in the response body — surface it instead of a bare "HTTP 400".
        try:
            body = e.read().decode('utf-8', 'replace')
        except Exception:
            body = ''
        logger.error(f"ComfyUI /prompt rejected the workflow ({e.code}): {body}")
        raise Exception(f"ComfyUI rejected the workflow ({e.code}). Detail: {body[:1000]}")

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
            elif message['type'] == 'execution_error':
                d = message.get('data', {})
                if d.get('prompt_id') == prompt_id:
                    raise Exception(
                        f"ComfyUI execution error in node {d.get('node_id')} "
                        f"({d.get('node_type')}): {d.get('exception_message')}"
                    )
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                # bytes 객체를 base64로 인코딩하여 JSON 직렬화 가능하게 변환
                if isinstance(image_data, bytes):
                    import base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

# 새 워크플로우 파일명: 이미지 개수별
_WORKFLOW_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow")
_WORKFLOW_FILES = {
    1: "qwen_image_edit_1_1image.json",
    2: "qwen_image_edit_1_2image.json",
    3: "qwen_image_edit_1_3image.json",
}

# 워크플로우별 노드 ID (이미지 개수에 따라 사용)
# 1-image: LoadImage=78, KSampler(seed)=3, prompt=111
# 2-image: 위 + LoadImage2=117
# 3-image: 위 + LoadImage3=119
_NODE_IMAGE_1 = "78"
_NODE_IMAGE_2 = "117"
_NODE_IMAGE_3 = "119"
_NODE_SEED = "3"
_NODE_PROMPT = "111"
_NODE_SAMPLING = "66"  # ModelSamplingAuraFlow — user LoRAs are chained in just before this
_NODE_MODEL = "37"     # UNETLoader — swap unet_name to use a custom Qwen-Image-Edit model
_NODE_WIDTH = "128"   # 현재 워크플로우에는 없음(선택 적용)
_NODE_HEIGHT = "129"  # 현재 워크플로우에는 없음(선택 적용)

# ------------------------------
# 입력 처리 유틸 (path/url/base64)
# ------------------------------
def process_input(input_data, temp_dir, output_filename, input_type):
    """입력 데이터를 처리하여 파일 경로를 반환하는 함수
    - input_type: "path" | "url" | "base64"
    """
    if input_type == "path":
        logger.info(f"📁 경로 입력 처리: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 URL 입력 처리: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("🔢 Base64 입력 처리")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"지원하지 않는 입력 타입: {input_type}")

def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ URL에서 파일을 성공적으로 다운로드했습니다: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget 다운로드 실패: {result.stderr}")
            raise Exception(f"URL 다운로드 실패: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ 다운로드 시간 초과")
        raise Exception("다운로드 시간 초과")
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        raise Exception(f"다운로드 중 오류 발생: {e}")

def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Base64 데이터를 파일로 저장하는 함수"""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 디코딩 실패: {e}")
        raise Exception(f"Base64 디코딩 실패: {e}")

def _models_dir(subdir):
    """Where to store a model/LoRA so ComfyUI can find it.
    Prefer the attached Network Volume (persists across workers); else the image dir.
    """
    vol = f"/runpod-volume/{subdir}"
    if os.path.isdir("/runpod-volume"):
        return vol
    return f"/ComfyUI/models/{subdir}"

def ensure_model_file(url_or_name, subdir):
    """Resolve a LoRA/model reference to a filename ComfyUI can load.
    - If it's a filename, return it as-is (must already be staged on the volume/image).
    - If it's an http(s) URL, download it (cached) into the volume/image and return
      the filename. Re-uses the file on subsequent requests/workers.
    """
    if not url_or_name:
        return None
    s = str(url_or_name).strip()
    if s.startswith("http://") or s.startswith("https://"):
        base = os.path.basename(urllib.parse.urlparse(s).path) or ""
        if not base.lower().endswith(".safetensors"):
            base = "dl_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:12] + ".safetensors"
        target_dir = _models_dir(subdir)
        os.makedirs(target_dir, exist_ok=True)
        dest = os.path.join(target_dir, base)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info(f"♻️  Reusing cached {subdir}: {dest}")
        else:
            logger.info(f"⬇️  Downloading {subdir} from {s} -> {dest}")
            download_file_from_url(s, dest)
        return base
    return s  # already a filename staged on the volume/image

# ============================================================================
# Reusable graph-injection helpers (shared by the legacy single-edit path and
# the per-person planned-edit loop). Each mutates the ComfyUI `prompt` graph.
# ============================================================================
def inject_model(prompt, model_ref):
    """Swap node 37's diffusion model. Raises on download/resolve failure."""
    if not model_ref or _NODE_MODEL not in prompt:
        return None
    fname = ensure_model_file(model_ref, "diffusion_models")
    if fname:
        prompt[_NODE_MODEL]["inputs"]["unet_name"] = fname
        logger.info(f"🧠 Using custom base model: {fname}")
    return fname


def _normalize_loras(loras, lora, lora_strength=1.0):
    """Accept either the `loras` array or the single `lora`/`lora_strength` shorthand."""
    if not loras and lora:
        return [{"name": lora, "strength": lora_strength}]
    return loras or []


def inject_loras(prompt, loras):
    """Stack user LoRAs on top of the existing chain (keeps the built-in Lightning
    LoRA) by inserting LoraLoaderModelOnly nodes right before ModelSamplingAuraFlow.
    Returns the number applied. Raises on a download/resolve failure."""
    if not loras:
        return 0
    chain_target = _NODE_SAMPLING  # ModelSamplingAuraFlow consumes the model
    if chain_target not in prompt or "model" not in prompt[chain_target].get("inputs", {}):
        logger.warning(f"Could not find node {_NODE_SAMPLING} to attach LoRAs; skipping LoRAs.")
        return 0
    prev = prompt[chain_target]["inputs"]["model"]  # e.g. ["89", 0]
    next_id = 9001
    applied = 0
    for lora in loras:
        if isinstance(lora, dict):
            ref = lora.get("name") or lora.get("url")
            strength_raw = lora.get("strength", 1.0)
        else:
            ref = lora
            strength_raw = 1.0
        if not ref:
            continue
        name = ensure_model_file(ref, "loras")  # may raise → caller converts to error
        if not name:
            continue
        try:
            strength = float(strength_raw)
        except (TypeError, ValueError):
            strength = 1.0
        node_id = str(next_id)
        next_id += 1
        prompt[node_id] = {
            "inputs": {"lora_name": name, "strength_model": strength, "model": prev},
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": f"User LoRA: {name}"},
        }
        prev = [node_id, 0]
        applied += 1
    prompt[chain_target]["inputs"]["model"] = prev
    if applied:
        logger.info(f"🎨 Applied {applied} user LoRA(s)")
    return applied


def connect_ws():
    """Wait for ComfyUI's HTTP server, then open a websocket. Returns the socket."""
    http_url = f"http://{server_address}:8188/"
    for http_attempt in range(180):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            if http_attempt == 179:
                raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            logger.warning(f"HTTP 연결 실패 (시도 {http_attempt+1}/180): {e}")
            time.sleep(1)

    ws = websocket.WebSocket()
    for attempt in range(int(180 / 5)):
        try:
            ws.connect(f"ws://{server_address}:8188/ws?clientId={client_id}")
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            return ws
        except Exception as e:
            if attempt == int(180 / 5) - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}): {e}")
            time.sleep(5)


def run_comfy_single_image(image_path, prompt_text, seed=None, model_ref=None,
                           loras=None, width=None, height=None):
    """Run the 1-image ComfyUI edit workflow once and return the result as base64.
    Used per-person by the planned loop (and trivially reusable elsewhere)."""
    workflow_path = os.path.join(_WORKFLOW_BASE, _WORKFLOW_FILES[1])
    prompt = load_workflow(workflow_path)

    prompt[_NODE_IMAGE_1]["inputs"]["image"] = image_path
    prompt[_NODE_PROMPT]["inputs"]["prompt"] = prompt_text or ""
    if _NODE_SEED in prompt and seed is not None:
        prompt[_NODE_SEED]["inputs"]["seed"] = seed
    if _NODE_WIDTH in prompt and width:
        prompt[_NODE_WIDTH]["inputs"]["value"] = width
    if _NODE_HEIGHT in prompt and height:
        prompt[_NODE_HEIGHT]["inputs"]["value"] = height

    inject_model(prompt, model_ref)
    inject_loras(prompt, loras)

    ws = connect_ws()
    try:
        images = get_images(ws, prompt)
    finally:
        ws.close()
    for node_id in images:
        if images[node_id]:
            return images[node_id][0]
    raise Exception("ComfyUI returned no image for this pass.")


# ============================================================================
# Planned per-subject pipeline: VLM planner → per-person iterative edit loop →
# region compositing → upload. See qwen-edit-aging-pipeline-spec.md and PLANNER.md.
# Triggered when the request sets pipeline="planned" or supplies a master_prompt.
# ============================================================================
def run_planned_edit(job_input, task_id):
    import planner
    import compositing as comp

    # --- collect the single input image (path / url / base64) -----------------
    if "image_path" in job_input:
        src = process_input(job_input["image_path"], task_id, "input_image_1.jpg", "path")
    elif "image_url" in job_input:
        src = process_input(job_input["image_url"], task_id, "input_image_1.jpg", "url")
    elif "image_base64" in job_input:
        src = process_input(job_input["image_base64"], task_id, "input_image_1.jpg", "base64")
    else:
        return {"error": "planned pipeline needs one image (image_path / image_url / image_base64)."}

    # Work in the editor's ~1MP space so planner bboxes line up with rendered output.
    original = comp.load_image(src)
    work_img = comp.scale_to_megapixels(original, 1.0)
    work_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), task_id))
    os.makedirs(work_dir, exist_ok=True)
    work_path = os.path.join(work_dir, "work.png")
    work_img.save(work_path)

    master_prompt = job_input.get("master_prompt") or planner.DEFAULT_MASTER_PROMPT
    value_range = job_input.get("value_range") or job_input.get("age_range")
    lora_options = job_input.get("lora_options") or {}
    global_loras = _normalize_loras(job_input.get("loras"), job_input.get("lora"),
                                    job_input.get("lora_strength", 1.0))
    model_ref = job_input.get("model_name") or job_input.get("model_url") or job_input.get("model")
    seed = job_input.get("seed")
    max_people = int(job_input.get("max_people", 12))
    edit_mode = job_input.get("edit_mode", "iterative")
    temperature = float(job_input.get("planner_temperature", 0.9))
    # How a per-person directive becomes the editor instruction. {directive} is required.
    template = job_input.get("edit_instruction_template", "Apply this change to this person: {directive}")

    # --- 1) plan ---------------------------------------------------------------
    try:
        people = planner.plan(
            work_img, master_prompt=master_prompt, value_range=value_range,
            lora_options=lora_options, seed=seed, temperature=temperature,
            max_people=max_people,
        )
    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return {"error": f"Planner stage failed: {e}"}
    logger.info(f"🗺️  Planner produced {len(people)} person directive(s).")

    def loras_for(person):
        """Per-person LoRAs = the planner-selected labeled option (if any) + globals."""
        chosen = []
        label = person.get("lora_label")
        if label and label in lora_options:
            opt = lora_options[label]
            chosen = opt if isinstance(opt, list) else [opt]
        return [*chosen, *global_loras]

    # --- 2) edit ---------------------------------------------------------------
    if edit_mode == "single_pass":
        # Cheaper, less reliable: one pass, all directives concatenated by position.
        combined = "; ".join(
            f"Person {p['person_id']} (left-to-right): {p['directive']}" for p in people
        )
        instruction = template.format(directive=combined) if "{directive}" in template else combined
        # Single-pass can only use global LoRAs (no per-person split in one pass).
        try:
            out_b64 = run_comfy_single_image(
                work_path, instruction, seed=seed, model_ref=model_ref, loras=global_loras)
        except Exception as e:
            return {"error": f"Editor (single_pass) failed: {e}"}
        work_img = comp.b64_to_image(out_b64)
        for p in people:
            p["applied_seed"] = seed
    else:
        # Iterative (default): edit each person, composite ONLY their bbox back so
        # untouched people stay pixel-exact and drift can't accumulate (spec §7).
        for i, p in enumerate(people):
            directive = p["directive"]
            instruction = template.format(directive=directive) if "{directive}" in template else f"{template} {directive}"
            pass_seed = None if seed is None else int(seed) + i  # vary per person, still reproducible
            p["applied_seed"] = pass_seed
            try:
                out_b64 = run_comfy_single_image(
                    work_path, instruction, seed=pass_seed, model_ref=model_ref,
                    loras=loras_for(p))
            except Exception as e:
                return {"error": f"Editor failed on person {p['person_id']}: {e}"}
            edited = comp.b64_to_image(out_b64)
            work_img = comp.composite(work_img, edited, p["bbox"], feather=int(job_input.get("feather", 6)))
            work_img.save(work_path)  # feed the composite into the next pass

    # --- 3) return (caller may also upload to a volume/S3) --------------------
    return {"image": comp.image_to_b64(work_img), "plan": people}


def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    # ------------------------------
    # Planned per-subject pipeline (VLM planner → per-person edit loop → composite).
    # Opt in with pipeline="planned", or implicitly by supplying a master_prompt.
    # Everything below this branch is the original single-instruction edit path,
    # left untouched so existing callers (batch UI, gallery, etc.) keep working.
    # ------------------------------
    if job_input.get("pipeline") == "planned" or job_input.get("master_prompt"):
        return run_planned_edit(job_input, task_id)

    # ------------------------------
    # 이미지 입력 수집 (1개 / 2개 / 3개)
    # 지원 키: image_path | image_url | image_base64
    #         image_path_2 | image_url_2 | image_base64_2
    #         image_path_3 | image_url_3 | image_base64_3
    # ------------------------------
    image_paths = []

    for i, suffix in enumerate([ "", "_2", "_3" ], start=1):
        path_key = f"image_path{suffix}"
        url_key = f"image_url{suffix}"
        b64_key = f"image_base64{suffix}"
        fname = f"input_image_{i}.jpg"
        if path_key in job_input:
            image_paths.append(process_input(job_input[path_key], task_id, fname, "path"))
        elif url_key in job_input:
            image_paths.append(process_input(job_input[url_key], task_id, fname, "url"))
        elif b64_key in job_input:
            image_paths.append(process_input(job_input[b64_key], task_id, fname, "base64"))
        else:
            break

    num_images = len(image_paths)
    if num_images == 0:
        return {"error": "최소 1개의 이미지 입력이 필요합니다. (image_path / image_url / image_base64 중 하나)"}

    if num_images not in _WORKFLOW_FILES:
        return {"error": f"지원하는 이미지 개수는 1, 2, 3개입니다. 입력된 이미지: {num_images}개"}

    workflow_filename = _WORKFLOW_FILES[num_images]
    workflow_path = os.path.join(_WORKFLOW_BASE, workflow_filename)
    if not os.path.exists(workflow_path):
        return {"error": f"워크플로우 파일을 찾을 수 없습니다: {workflow_path}"}

    prompt = load_workflow(workflow_path)

    # 노드 번호는 각 워크플로우 JSON과 동일하게 사용
    prompt[_NODE_IMAGE_1]["inputs"]["image"] = image_paths[0]
    if num_images >= 2:
        prompt[_NODE_IMAGE_2]["inputs"]["image"] = image_paths[1]
    if num_images >= 3:
        prompt[_NODE_IMAGE_3]["inputs"]["image"] = image_paths[2]

    prompt[_NODE_PROMPT]["inputs"]["prompt"] = job_input.get("prompt", "")
    if _NODE_SEED in prompt and "seed" in job_input:
        prompt[_NODE_SEED]["inputs"]["seed"] = job_input["seed"]
    if _NODE_WIDTH in prompt and "width" in job_input:
        prompt[_NODE_WIDTH]["inputs"]["value"] = job_input["width"]
    if _NODE_HEIGHT in prompt and "height" in job_input:
        prompt[_NODE_HEIGHT]["inputs"]["value"] = job_input["height"]

    # ------------------------------
    # Custom base model + LoRAs (optional). See inject_model / inject_loras above
    # for the graph wiring. Model/LoRA refs may be a filename staged on the volume
    # (or image) or a URL that's downloaded + cached onto the volume on first use.
    #   "model_name": "my_qwen_finetune.safetensors"  | "model_url": "https://.../m.safetensors"
    #   "loras": [ {"name": "x.safetensors", "strength": 1.0}, {"url": "...", "strength": 0.8} ]
    #   "lora": "x.safetensors", "lora_strength": 1.0   (single convenience)
    # ------------------------------
    model_ref = job_input.get("model_name") or job_input.get("model_url") or job_input.get("model")
    try:
        inject_model(prompt, model_ref)
    except Exception as e:
        return {"error": f"Failed to load model '{model_ref}': {e}"}

    try:
        inject_loras(prompt, _normalize_loras(job_input.get("loras"), job_input.get("lora"),
                                              job_input.get("lora_strength", 1.0)))
    except Exception as e:
        return {"error": f"Failed to load LoRA: {e}"}

    ws = connect_ws()
    images = get_images(ws, prompt)
    ws.close()

    # 이미지가 없는 경우 처리
    if not images:
        return {"error": "이미지를 생성할 수 없습니다."}
    
    # 첫 번째 이미지 반환
    for node_id in images:
        if images[node_id]:
            return {"image": images[node_id][0]}
    
    return {"error": "이미지를 찾을 수 없습니다."}

runpod.serverless.start({"handler": handler})