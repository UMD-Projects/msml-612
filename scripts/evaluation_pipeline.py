# %%
# Set JAX_PLATFORMS=''
import os
# os.environ["JAX_PLATFORMS"] = "cpu"
from flaxdiff.inference.pipeline import DiffusionInferencePipeline

# %%
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from wandb import Image as wandbImage
import tqdm
import grain.python as pygrain
import torch
from flaxdiff.samplers.euler import EulerAncestralSampler
import numpy as np

class EvaluationPipeline(DiffusionInferencePipeline):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fid = self.FrechetInceptionDistance()
        self.kid = self.KernelInceptionDistance(subset_size=32)
        self.lpip = self.LearnedPerceptualImagePatchSimilarity()
        import wandb
        self.wandb = wandb.init(
            project='mlops-evaluations',
            entity='umd-projects',
            name=self.name,
            job_type='evaluation',
            config=self.config,
        ) 
        
    def update_metrics(self, original_images, generated_images):
        """
        Computes FID, KID, and LPIPS metrics between original and generated images.
        All images should be in the range [0, 255].
        """
        self.fid.update(original_images, real=True)  
        self.fid.update(generated_images, real=False)
        
        self.kid.update(original_images, real=True)
        self.kid.update(generated_images, real=False)
        
        # LPIPS expects images in the range [0, 1]
        original_images = original_images / 255.0
        generated_images = generated_images / 255.0
        self.lpip.update(original_images, generated_images)
    
    def evaluate(
        self, 
        dataloader: pygrain.DataLoader, 
        diffusion_steps=50, 
        sampler_class=EulerAncestralSampler,
        iterations=20,
        batch_size=8,
        image_size=256,
    ):
        iterator = iter(dataloader)
        for i in tqdm.tqdm(range(iterations)):
            batch = next(iterator)
            original_images = batch['image']
            conditions = batch['text']
            model_conditioning_inputs = self.input_config.conditions[0].encoder.encode_from_tokens(conditions)
            
            generated = self.generate_samples(
                num_samples=batch_size,
                resolution=image_size,
                diffusion_steps=diffusion_steps,
                guidance_scale=3.0,
                start_step=1000,
                sampler_class=sampler_class,
                conditioning_data_tokens=(model_conditioning_inputs,)
            )
            
            # Calculate metrics
            # Original images are in [0, 255]. Generated images are in [-1, 1]
            generated_255 = (generated * 127.5 + 127.5).astype(jnp.uint8)
            original_255 = original_images.astype(jnp.uint8)
            
            # Convert to torch tensors
            original_255 = torch.from_numpy(original_255).permute(0, 3, 1, 2)
            generated_255 = torch.from_numpy(np.array(generated_255)).permute(0, 3, 1, 2)
            
            self.update_metrics(original_255, generated_255)
            
            # Plot the images
            # Put them on wandb
            orig = original_255[0].permute(1, 2, 0).numpy()
            gen = generated_255[0].permute(1, 2, 0).numpy()
            self.wandb.log({
                "Original+Generated": wandbImage(
                    np.concatenate((orig, gen), axis=1),
                    caption=f"Original + Generated {i}"
                )
            })
        
        fid_score = self.fid.compute()
        kid_score = self.kid.compute()
        lpips_score = self.lpip.compute()
        
        metrics = {
            'FID': fid_score,
            'KID_mean': kid_score[0],
            'KID_std': kid_score[1],
            'LPIPS': lpips_score
        }
        
        # TODO: Push these metrics to wandb model registry
        self.wandb.log({"metrics": metrics})
        
        # push to registry
        self.push_to_registry()
            
            
        return metrics
    
    def push_to_registry(
        self,
        registry_name: str = 'wandb-registry-model',
        aliases = [],
    ):
        """
        Push the model to wandb registry.
        Args:
            registry_name: Name of the model registry.
            aliases: List of aliases for the model.
        """
        modelname = self.config['raw_config']['modelname']
        target_path = f"{registry_name}/final-{modelname}"
        
        self.wandb.link_artifact(
            artifact=self.artifact,
            target_path=target_path,
            aliases=aliases,
        )
        print(f"Model pushed to registry at {target_path}")
        return target_path

def evaluate_model(
    model_registry: str = "diffusion-laiona_coco-res256",
    version: str = 'best',
    batch_size: int = 128,
    image_size: int = 256,
    diffusion_steps: int = 200,
    iterations: int = 100,
):
    pipeline = EvaluationPipeline.from_wandb_registry(
        modelname=args.model_registry,
        project='mlops-msml605-project',
        entity='umd-projects',
        version=args.version,
    )

    from flaxdiff.data.dataloaders import get_dataset_grain
    data = get_dataset_grain(
        "laiona_coco",
        batch_size=batch_size,
        image_scale=image_size,
        dataset_source="/home/mrwhite0racle/gcs_mount",
        method=None
    )

    datalen = data['train_len']
    batches = datalen // batch_size

    pipeline.evaluate(
        dataloader=data['train'](),
        diffusion_steps=150,
        sampler_class=EulerAncestralSampler,
        iterations=iterations, 
        batch_size=batch_size, 
        image_size=image_size
    )

if __name__ == "__main__":
    # Parse the 'model registry' and 'version argument from the command line
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model from the registry.')
    parser.add_argument('--model_registry', type=str, help='Model registry name', default="diffusion-laiona_coco-res256")
    parser.add_argument('--version', type=str, help='Model version', default='best')
    args = parser.parse_args()
    
    evaluate_model(
        model_registry=args.model_registry,
        version=args.version,
        batch_size=128,
        image_size=256,
        diffusion_steps=200,
        iterations=100,
    )