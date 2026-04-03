import os
import uuid
import base64
import io
import threading
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np

from flaxdiff.inference.pipeline import DiffusionInferencePipeline
from flaxdiff.samplers.euler import EulerAncestralSampler
from flaxdiff.samplers.euler import EulerSampler
from flaxdiff.samplers.ddim import DDIMSampler
from flaxdiff.samplers.heun_sampler import HeunSampler
from flaxdiff.samplers.rk4_sampler import RK4Sampler
from flaxdiff.samplers.ddpm import DDPMSampler, SimpleDDPMSampler

app = FastAPI()
job_store = {}

# Preload the pipeline from wandb registry
pipeline_store = {
    "diffusion-laiona_coco-res256-sweep-wgjsicqf": DiffusionInferencePipeline.from_wandb_registry(
                    modelname="diffusion-laiona_coco-res256-sweep-wgjsicqf",
                    project="mlops-msml605-project",
                    entity="umd-projects",
                    #version=req.version
                    version="best"
                ),
}

SAMPLER_CLASSES = {
    "euler": EulerAncestralSampler,
    "euler_ancestral": EulerAncestralSampler,
    "euler_sampler": EulerSampler,
    "ddim": DDIMSampler,
    "heun": HeunSampler,
    "rk4": RK4Sampler,
    "ddpm": DDPMSampler,
    "simple_ddpm": SimpleDDPMSampler,
}
    

# Request schema
class GenerateRequest(BaseModel):
    prompts: List[str]
    model_name: Optional[str] = 'diffusion-laiona_coco-res256'
    #version: Optional[str] = "best"
    resolution: Optional[int] = 256
    diffusion_steps: Optional[int] = 200
    guidance_scale: Optional[float] = 3.0
    start_step: Optional[int] = 1000
    sampler_class: Optional[str] = "euler_ancestral"

@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "running", "result": None}

    def run_generation():
        try:
            pipeline = pipeline_store.get(req.model_name, None)
            if pipeline is None:
                print(f"[Job {job_id}] Loading pipeline for model: {req.model_name}")
                pipeline = DiffusionInferencePipeline.from_wandb_registry(
                    modelname=req.model_name,
                    project="mlops-msml605-project",
                    entity="umd-projects",
                    #version=req.version
                    version="best"
                )
                pipeline_store[req.model_name] = pipeline
                print(f"[Job {job_id}] Pipeline loaded for model: {req.model_name}")
            else:
                print(f"[Job {job_id}] Using cached pipeline for model: {req.model_name}")
                
            samples = pipeline.generate_samples(
                num_samples=len(req.prompts),
                resolution=req.resolution,
                diffusion_steps=req.diffusion_steps,
                guidance_scale=req.guidance_scale,
                start_step=req.start_step,
                sampler_class=SAMPLER_CLASSES.get(req.sampler_class, EulerAncestralSampler),
                conditioning_data=req.prompts
            )

            images_b64 = []
            for i, img in enumerate(samples):
                try:
                    img_np = np.array(img)
                    if img_np.ndim == 2:
                        img_np = np.stack([img_np] * 3, axis=-1)

                    img_np = np.nan_to_num(img_np, nan=0.0)
                    img_uint8 = np.clip((img_np * 127.5 + 127.5), 0, 255).astype(np.uint8)

                    buf = io.BytesIO()
                    Image.fromarray(img_uint8).save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode()
                    images_b64.append(img_b64)
                except Exception as e:
                    print(f"[Job {job_id}] Failed to process image {i}: {e}")



            print(f"[Job {job_id}] Number of images generated: {len(images_b64)}")
            job_store[job_id] = {"status": "completed", "result": images_b64}

        except Exception as e:
            job_store[job_id] = {"status": "failed", "error": str(e)}

    threading.Thread(target=run_generation).start()
    return {"job_id": job_id, "status": "running"}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)