# Per-subject randomized edit pipeline (VLM planner → editor)

This worker can run a **planned** pipeline on top of the normal single-instruction
edit. It externalizes the per-person reasoning into a **VLM planner** (Qwen3-VL)
that looks at the image, locates each person, and writes an individually-grounded,
**randomized** directive per person. The Qwen-Image-Edit editor then renders each
directive, and only the edited person's region is composited back so untouched
people stay pixel-exact.

The first use case is **aging**, but the architecture is general: swap the
`master_prompt` (and the LoRA options) to do other per-subject randomized edits to
groups of people. See `qwen-edit-aging-pipeline-spec.md` for the design rationale
(read §2 — the naive "age everyone by 30 years" global prompt is exactly what this
avoids).

## How it runs (single worker)

```
image ─▶ planner.plan()  ─▶ people[{bbox, directive, lora_label}] ─▶ per-person edit loop ─▶ composite ─▶ image+plan
        Qwen3-VL (transformers)                                      ComfyUI 1-image workflow   feathered bbox
```

Both stages live in **one** serverless worker (no network hop, one cold start).
The planner is **lazy-loaded** — nothing here touches the GPU unless a planned
request actually arrives, so the normal edit path is unaffected.

## Triggering it

Set `pipeline: "planned"` **or** just supply a `master_prompt`. Anything else uses
the original single-instruction path.

Use RunPod's **async** `/run` + `/status` (the consumer app already does): an
iterative multi-person edit is N diffusion passes and will exceed the 90s sync
window.

## Request contract

```jsonc
{
  "input": {
    "pipeline": "planned",
    "image_base64": "<...>",            // or image_url / image_path (ONE image)

    "master_prompt": "<planner meta-instruction>",   // defaults to the aging prompt
    "value_range": [15, 45],            // alias: age_range — the range the planner randomizes within
    "edit_mode": "iterative",           // "iterative" (default) | "single_pass"
    "seed": 12345,                      // optional; reproducible randomness (per-person = seed+i)
    "max_people": 12,
    "planner_temperature": 0.9,         // >0 → directives vary per person
    "feather": 6,                       // bbox mask feather (px)

    // How a directive becomes the editor prompt. {directive} is substituted.
    "edit_instruction_template": "Age this person realistically: {directive}",

    // Per-subject LoRA selection: the planner picks a label per person; the handler
    // injects that LoRA for that person's pass. Values are a LoRA spec or list of them.
    "lora_options": {
      "gray_hair":  { "name": "gray_hair.safetensors", "strength": 0.6 },
      "wrinkles":   { "url": "https://.../wrinkles.safetensors", "strength": 0.7 }
    },

    // Global LoRAs applied to EVERY pass (stacked on top of any per-person one).
    "loras": [ { "name": "skin_realism.safetensors", "strength": 0.4 } ],

    // Optional base-model swap (Qwen-Image-Edit compatible).
    "model_name": "my_qwen_finetune.safetensors"
  }
}
```

### Response

```jsonc
{
  "image": "<base64 PNG of the final composited image>",
  "plan": [
    { "person_id": 1, "bbox": [x, y, w, h], "estimated_age": 34,
      "change_amount": 26, "directive": "gray the temples, deepen nasolabial folds...",
      "lora_label": "gray_hair", "applied_seed": 12345 }
  ]
}
```

The `plan` is logged/returned for reproducibility and auditing.

## Generalizing beyond aging

1. Replace `master_prompt` with a meta-instruction for the new edit (e.g. "give
   each person a different era-appropriate outfit", "apply a distinct weathering
   level"). Keep the "vary it per person / be consistent with current appearance"
   framing — that's what produces variety instead of uniformity.
2. Define `lora_options` with the LoRAs that render those different change types,
   and tell the planner (in the master prompt) to choose `lora_label` per person.
3. Set `edit_instruction_template` so the directive reads naturally for the editor.

## Planner model — verify before relying on it

- The VLM is **pulled from Hugging Face at runtime** on the first planned request and
  cached on the worker's local disk. So the worker needs **no Network Volume and isn't
  pinned to a region**, and the image stays small (baking a 16GB model blew the 30-min
  hub build limit — that's why it's not baked).
  - Default `PLANNER_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct` (~8GB, plain bf16) keeps the
    one-time per-worker download light. **Change the model with NO rebuild** by setting
    `PLANNER_MODEL_ID` on the endpoint (e.g. `Qwen/Qwen3-VL-8B-Instruct` for more
    capability, at a bigger download).
  - Use **min-workers = 1** during a testing session so the download happens once and
    the worker stays warm; otherwise every fresh worker re-downloads it.
  - Want zero per-worker download? Either bake it into the image (fast cold starts but
    fights the build limit) or stage it on a Network Volume (fast, but re-pins region).
- **Confirm** the installed `transformers` supports the chosen Qwen3-VL revision. The
  loader tries `AutoModelForImageTextToText` then a few known class names.
- **Instruct** edition for speed; switch to a Thinking edition only if directive
  quality is weak (costs latency).

## Open decisions (operator)

- **GPU tier / VRAM** — editor alone fits 24GB; planner + editor co-resident wants
  ~40–48GB (L40S / A6000) for headroom. `.runpod/hub.json` now lists 48GB first.
  Confirm against your chosen VLM size (8B vs 32B).
- **executionTimeout / ttl** — raise them in the RunPod endpoint settings; iterative
  multi-person edits are several passes (well under any hard limit, but not sub-90s).
- **transformers pinning** — upgrading transformers for Qwen3-VL may affect ComfyUI
  custom nodes. If a node breaks, pin a version compatible with both, or split the
  planner into its own endpoint.
- **bbox vs segmentation** — compositing uses feathered bboxes (simplest). If edits
  bleed at region edges on overlapping people, add a segmentation step (e.g. SAM)
  for tight masks.
- **Cost** — iterative mode = N passes for N people, so GPU-seconds scale with crowd
  size. `single_pass` is the cheaper, lower-fidelity lever for large batches.
