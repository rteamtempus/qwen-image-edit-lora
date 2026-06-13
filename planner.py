"""
VLM planner stage for the per-subject randomized edit pipeline.

Why this exists (read qwen-edit-aging-pipeline-spec.md §2): Qwen-Image-Edit is a
diffusion editor with weak high-level reasoning. A single global instruction
("age everyone by 30 years") collapses every face toward the *average* of that
concept — same gray hair, same wrinkles. The variety + per-individual grounding
have to come from a VLM "planner" that looks at the image, locates each person,
and writes an individually-grounded, *randomized* directive per person. The
editor only renders what the planner decides.

This module is intentionally decoupled from the editor (ComfyUI) so the rest of
the worker is unaffected when the planned pipeline isn't used. It is lazy-loaded:
nothing here touches the GPU until plan() is first called.

⚠️ VERIFY AT BUILD TIME (the spec calls this out explicitly, §6):
  - The exact Qwen3-VL repo id and the transformers auto-class depend on the
    transformers version installed in the worker image. We default to
    AutoModelForImageTextToText (the current image-text-to-text auto class) and
    fall back across a couple of known class names. Override the repo id with the
    PLANNER_MODEL_ID env var; stage the weights on the network volume (see
    PLANNER.md) so cold starts don't re-download tens of GB.
"""

import os
import re
import json
import logging

logger = logging.getLogger(__name__)

# Repo id (or local path) of the planner VLM. Pulled from HF at runtime on first
# use and cached on the worker's local disk (no Network Volume, region-free). The
# Dockerfile sets PLANNER_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct; override it on the
# endpoint to use a bigger model (e.g. Qwen/Qwen3-VL-8B-Instruct) with NO rebuild.
# See PLANNER.md.
PLANNER_MODEL_ID = os.getenv("PLANNER_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
PLANNER_DEVICE = os.getenv("PLANNER_DEVICE", "cuda")
# Max new tokens for the plan JSON. A dozen people with directives fits well under this.
PLANNER_MAX_NEW_TOKENS = int(os.getenv("PLANNER_MAX_NEW_TOKENS", "1024"))

# The default master prompt is the aging use case (the first target). It is a
# generic, reusable meta-instruction: swapping it makes the pipeline do other
# per-subject randomized edits. The operator overrides it per request.
DEFAULT_MASTER_PROMPT = (
    "You are planning a per-person edit for an image containing one or more people.\n"
    "For EACH person, independently:\n"
    "  1. Locate them and return a bounding box [x, y, w, h] in PIXEL coords of THIS image.\n"
    "  2. Estimate their current apparent age.\n"
    "  3. Choose a randomized aging amount within the requested range, varying it per person.\n"
    "  4. Write a 'directive' that is CONSISTENT WITH THIS PERSON'S CURRENT APPEARANCE\n"
    "     (existing hair color/length, facial hair, skin tone, face shape, build) and that\n"
    "     differs in STYLE from the other people in the image. Draw from varied aging axes:\n"
    "       - hair: graying / salt-and-pepper / thinning / receding / whitening\n"
    "       - skin: forehead lines / crow's feet / nasolabial folds / age spots / sagging / weathering\n"
    "       - structure: softened jawline / jowls / hollowing / brow descent\n"
    "     Do not assign features that conflict with current appearance (e.g. don't add a\n"
    "     gray beard to a clean-shaven person unless aging into one realistically)."
)

# Appended automatically so the model returns parseable JSON. {lora_clause} is
# filled in when the request supplies lora_options (per-subject LoRA selection).
_JSON_INSTRUCTION = (
    "\n\nReturn STRICT JSON ONLY — no prose, no markdown fences — matching this schema:\n"
    '{{ "people": [ {{ "bbox":[x,y,w,h], "estimated_age":int, "change_amount":int,\n'
    '                "directive":"..."{lora_clause} }} ] }}\n'
    "Coordinates are pixels in the provided image. Order people left-to-right.{range_clause}"
)

# Lazy singletons — populated on first plan() call, reused across warm invocations.
_processor = None
_model = None
_torch = None


def _load():
    """Load the VLM once per worker. Heavy; only called when a planned edit runs."""
    global _processor, _model, _torch
    if _model is not None:
        return
    import torch
    _torch = torch
    from transformers import AutoProcessor

    logger.info(f"🧠 Loading planner VLM: {PLANNER_MODEL_ID}")
    _processor = AutoProcessor.from_pretrained(PLANNER_MODEL_ID, trust_remote_code=True)

    # The auto-class name varies by transformers version / model generation.
    # Try the current general class first, then older Qwen-VL-specific ones.
    last_err = None
    for class_name in (
        "AutoModelForImageTextToText",
        "Qwen3VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForVision2Seq",
    ):
        try:
            import transformers
            cls = getattr(transformers, class_name)
        except AttributeError:
            continue
        try:
            _model = cls.from_pretrained(
                PLANNER_MODEL_ID,
                torch_dtype="auto",
                device_map=PLANNER_DEVICE if PLANNER_DEVICE != "cpu" else None,
                trust_remote_code=True,
            )
            logger.info(f"🧠 Planner loaded via {class_name}")
            break
        except Exception as e:  # noqa: BLE001 — try the next candidate class
            last_err = e
            logger.warning(f"Planner load via {class_name} failed: {e}")
    if _model is None:
        raise RuntimeError(
            f"Could not load planner VLM '{PLANNER_MODEL_ID}'. Verify the repo id and "
            f"that the installed transformers supports it. Last error: {last_err}"
        )
    _model.eval()


def _strip_fences(raw: str) -> str:
    """Remove ```json ... ``` fences and grab the outermost JSON object if wrapped in prose."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # If the model still wrapped JSON in prose, extract the first {...} block.
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    return s.strip()


def _build_prompt(master_prompt: str, value_range, lora_options) -> str:
    """Assemble the full planner instruction from the master prompt + runtime params."""
    lora_clause = ""
    if lora_options:
        labels = ", ".join(f'"{k}"' for k in lora_options.keys())
        lora_clause = (
            ',\n                "lora_label":"<one of: ' + labels + ', or empty>"'
        )
    range_clause = ""
    if value_range and len(value_range) == 2:
        range_clause = (
            f" The requested change range is {value_range[0]}..{value_range[1]}; "
            f"pick a randomized 'change_amount' within it, varied per person."
        )
    if lora_options:
        range_clause += (
            " Choose 'lora_label' per person to best match the change you describe; "
            "leave it empty if none fits."
        )
    instruction = _JSON_INSTRUCTION.format(lora_clause=lora_clause, range_clause=range_clause)
    return (master_prompt or DEFAULT_MASTER_PROMPT).strip() + instruction


def _run_vlm(image, full_prompt: str, seed=None, temperature: float = 0.9) -> str:
    """Single VLM forward pass → raw text. temperature>0 gives per-person variety."""
    if seed is not None:
        try:
            _torch.manual_seed(int(seed))
        except (TypeError, ValueError):
            pass

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt},
            ],
        }
    ]
    # Two transformers conventions exist for Qwen-VL. Newer versions tokenize the
    # image inline via apply_chat_template(return_dict=True); older ones need a
    # separate processor() call with images extracted by qwen_vl_utils. Try the
    # combined path first, fall back if no pixel values came through.
    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" not in inputs:
        text = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
        except Exception:
            image_inputs = [image]
        inputs = _processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    inputs = {k: (v.to(_model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    do_sample = temperature and temperature > 0
    gen = _model.generate(
        **inputs,
        max_new_tokens=PLANNER_MAX_NEW_TOKENS,
        do_sample=bool(do_sample),
        temperature=float(temperature) if do_sample else None,
        top_p=0.95 if do_sample else None,
    )
    # Drop the prompt tokens; decode only the newly generated continuation.
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    text = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return text


def _normalize_people(parsed, lora_options):
    """Coerce the parsed JSON into a stable list of person dicts.

    Accepts the spec's aging-specific keys (aging_directive / aging_years) as well
    as the generic ones (directive / change_amount) so older master prompts still work.
    """
    people = parsed.get("people") if isinstance(parsed, dict) else None
    if not isinstance(people, list):
        raise ValueError("planner JSON missing a 'people' array")

    valid_labels = set(lora_options.keys()) if lora_options else set()
    out = []
    for idx, p in enumerate(people):
        if not isinstance(p, dict):
            continue
        bbox = p.get("bbox") or p.get("box")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            logger.warning(f"Skipping person {idx}: bad bbox {bbox!r}")
            continue
        directive = p.get("directive") or p.get("aging_directive") or ""
        if not str(directive).strip():
            logger.warning(f"Skipping person {idx}: empty directive")
            continue
        label = p.get("lora_label") or ""
        if label and valid_labels and label not in valid_labels:
            logger.warning(f"Person {idx}: unknown lora_label {label!r}; ignoring")
            label = ""
        out.append({
            "person_id": idx + 1,
            "bbox": [float(v) for v in bbox],
            "estimated_age": p.get("estimated_age"),
            "change_amount": p.get("change_amount", p.get("aging_years")),
            "directive": str(directive).strip(),
            "lora_label": label,
        })
    if not out:
        raise ValueError("planner returned no usable people")
    return out


def plan(image, master_prompt=None, value_range=None, lora_options=None,
         seed=None, temperature=0.9, max_people=12):
    """Run the planner on a PIL image and return a list of per-person directives.

    Returns: list[{person_id, bbox:[x,y,w,h], estimated_age, change_amount,
                   directive, lora_label}]
    Raises on unrecoverable parse failure (after one corrective retry).
    """
    _load()
    full_prompt = _build_prompt(master_prompt, value_range, lora_options)

    raw = _run_vlm(image, full_prompt, seed=seed, temperature=temperature)
    try:
        parsed = json.loads(_strip_fences(raw))
        people = _normalize_people(parsed, lora_options)
    except Exception as first_err:  # noqa: BLE001
        # One corrective retry — the spec mandates exactly one (§6).
        logger.warning(f"Planner JSON parse failed ({first_err}); retrying once.")
        corrective = full_prompt + (
            "\n\nYour previous reply was not valid JSON for the schema. "
            "Reply again with STRICT JSON ONLY, no prose, no fences."
        )
        # Nudge the seed so the retry isn't a verbatim repeat.
        retry_seed = None if seed is None else int(seed) + 1
        raw = _run_vlm(image, corrective, seed=retry_seed, temperature=temperature)
        parsed = json.loads(_strip_fences(raw))  # let a 2nd failure propagate
        people = _normalize_people(parsed, lora_options)

    return people[: max_people or 12]
