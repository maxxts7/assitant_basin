"""
Basin Experiment: Perturbation-recovery protocol for establishing
the existence of an assistant basin in transformer activation space.

Uses the assistant-axis infrastructure (Lu et al., 2026) for model loading,
activation extraction, and hook-based intervention.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from assistant_axis.internals.model import ProbingModel
from assistant_axis.steering import ActivationSteering
from assistant_axis.axis import load_axis, project


# ---------------------------------------------------------------------------
# Model & axis configuration per supported model
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "google/gemma-2-27b-it": {
        "hf_axis_file": "gemma-2-27b/assistant_axis.pt",
        "target_layer": 22,
        "num_layers": 46,
    },
    "Qwen/Qwen3-32B": {
        "hf_axis_file": "qwen-3-32b/assistant_axis.pt",
        "target_layer": 32,
        "num_layers": 64,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "hf_axis_file": "llama-3.3-70b/assistant_axis.pt",
        "target_layer": 40,
        "num_layers": 80,
    },
}

HF_AXIS_REPO = "lu-christina/assistant-axis-vectors"


def download_axis(model_name: str, cache_dir: str = "results") -> str:
    """Download pre-computed assistant axis from HuggingFace."""
    cfg = MODEL_CONFIGS[model_name]
    return hf_hub_download(
        repo_id=HF_AXIS_REPO,
        filename=cfg["hf_axis_file"],
        repo_type="dataset",
        local_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def random_unit_vector(dim: int, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    """Sample a random unit vector in R^dim."""
    v = torch.randn(dim, device=device, dtype=dtype)
    return v / v.norm()


def make_perturbation(
    h_baseline: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Create a perturbation delta = alpha * ||h_baseline|| * direction."""
    magnitude = alpha * h_baseline.norm().item()
    return magnitude * direction.to(h_baseline.dtype)


# ---------------------------------------------------------------------------
# Core experiment class
# ---------------------------------------------------------------------------

class BasinExperiment:
    """Runs perturbation-recovery experiments on transformer residual streams.

    Uses ProbingModel for model loading and ActivationSteering for hook-based
    perturbation injection. Measures whether subsequent layers restore
    activations toward the baseline after perturbation.
    """

    def __init__(
        self,
        model_name: str,
        axis_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        deterministic: bool = False,
    ):
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # NOTE: Do NOT disable flash_sdp/mem_efficient_sdp here.
            # The math SDPA fallback overflows in bf16 on large models.
            # Instead, we pass attn_implementation="eager" to the model.

        self.model_name = model_name
        self._deterministic = deterministic

        if deterministic:
            # Load model ourselves with eager attention, then wrap in ProbingModel
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="eager",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pm = ProbingModel.from_existing(model, tokenizer, model_name=model_name)
        else:
            self.pm = ProbingModel(model_name, device=device, dtype=dtype)
        self.tokenizer = self.pm.tokenizer
        self.layers = self.pm.get_layers()
        self.num_layers = len(self.layers)

        # Load axis
        if axis_path is None:
            axis_path = download_axis(model_name)
        self.axis = load_axis(axis_path)  # (num_layers, hidden_dim)
        self.hidden_dim = self.axis.shape[-1]

    # ------------------------------------------------------------------
    # Trajectory extraction
    # ------------------------------------------------------------------

    def get_baseline_trajectory(
        self, input_ids: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run unperturbed forward pass and cache residual stream at all layers.

        Args:
            input_ids: tokenized prompt, shape (1, seq_len), on model device.

        Returns:
            activations: {layer_idx: tensor of shape (hidden_dim,)} at last token.
            logits: final logits tensor of shape (vocab_size,).
        """
        activations = {}
        handles = []

        for layer_idx in range(self.num_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    act = output[0] if isinstance(output, tuple) else output
                    activations[idx] = act[0, -1, :].detach().clone().cpu().float()
                return hook_fn
            handles.append(self.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        with torch.inference_mode():
            outputs = self.pm.model(input_ids)

        for h in handles:
            h.remove()

        logits = outputs.logits[0, -1, :].detach().clone().cpu().float()
        return activations, logits

    def get_perturbed_trajectory(
        self,
        input_ids: torch.Tensor,
        perturb_layer: int,
        delta: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run forward pass with perturbation injected at perturb_layer.

        Args:
            input_ids: tokenized prompt, shape (1, seq_len), on model device.
            perturb_layer: which layer to inject the perturbation at.
            delta: perturbation vector, shape (hidden_dim,).

        Returns:
            activations: {layer_idx: tensor} for all layers AFTER perturb_layer.
            logits: final logits tensor of shape (vocab_size,).
        """
        captured = {}
        handles = []

        # Register extraction hooks on all downstream layers
        for layer_idx in range(perturb_layer + 1, self.num_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    act = output[0] if isinstance(output, tuple) else output
                    captured[idx] = act[0, -1, :].detach().clone().cpu().float()
                return hook_fn
            handles.append(self.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        # Inject perturbation at perturb_layer via ActivationSteering
        delta_device = delta.to(self.pm.model.dtype)
        with ActivationSteering(
            self.pm.model,
            steering_vectors=[delta_device],
            coefficients=[1.0],
            layer_indices=[perturb_layer],
            intervention_type="addition",
            positions="last",
        ):
            with torch.inference_mode():
                outputs = self.pm.model(input_ids)

        for h in handles:
            h.remove()

        logits = outputs.logits[0, -1, :].detach().clone().cpu().float()
        return captured, logits

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_recovery_metrics(
        self,
        baseline_acts: dict[int, torch.Tensor],
        perturbed_acts: dict[int, torch.Tensor],
        baseline_logits: torch.Tensor,
        perturbed_logits: torch.Tensor,
        perturb_layer: int,
    ) -> list[dict]:
        """Compute recovery metrics at each layer downstream of the perturbation.

        Returns a list of dicts, one per downstream layer, with:
            - downstream_layer
            - normalized_distance
            - cosine_similarity
            - axis_projection_gap
            - axis_projection_baseline
            - axis_projection_perturbed
        Plus one entry for the final output:
            - kl_divergence
            - top1_preserved
        """
        records = []

        for l in range(perturb_layer + 1, self.num_layers):
            if l not in baseline_acts or l not in perturbed_acts:
                continue

            b = baseline_acts[l]
            p = perturbed_acts[l]
            diff = p - b

            # Normalized L2 distance
            norm_dist = diff.norm().item() / (b.norm().item() + 1e-12)

            # Cosine similarity
            cos_sim = F.cosine_similarity(b.unsqueeze(0), p.unsqueeze(0)).item()

            # Projection onto assistant axis at this layer
            axis_l = self.axis[l].float()
            axis_l_norm = axis_l / (axis_l.norm() + 1e-12)
            proj_b = (b @ axis_l_norm).item()
            proj_p = (p @ axis_l_norm).item()
            axis_gap = proj_b - proj_p  # positive = perturbed is further from assistant

            records.append({
                "downstream_layer": l,
                "normalized_distance": norm_dist,
                "cosine_similarity": cos_sim,
                "axis_projection_gap": axis_gap,
                "axis_projection_baseline": proj_b,
                "axis_projection_perturbed": proj_p,
            })

        # Output-level metrics
        log_p_perturbed = F.log_softmax(perturbed_logits, dim=-1)
        p_baseline = F.softmax(baseline_logits, dim=-1)
        kl = F.kl_div(log_p_perturbed, p_baseline, reduction="sum").item()
        top1_match = (baseline_logits.argmax() == perturbed_logits.argmax()).item()

        # Attach output metrics to every record
        for r in records:
            r["kl_divergence"] = kl
            r["top1_preserved"] = top1_match

        return records

    # ------------------------------------------------------------------
    # Tokenization helper
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize a prompt with chat template applied, return input_ids on device."""
        conversation = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(text, return_tensors="pt")
        # With device_map="auto", model has no single .device — find the embedding device
        if hasattr(self.pm.model, "hf_device_map"):
            first_device = next(iter(self.pm.model.hf_device_map.values()))
            device = f"cuda:{first_device}" if isinstance(first_device, int) else first_device
        else:
            device = self.pm.model.device
        return tokens["input_ids"].to(device)

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------

    def run_single_prompt(
        self,
        prompt: str,
        perturb_layers: list[int],
        alphas: list[float],
        n_random_dirs: int = 5,
        seed: int = 42,
    ) -> list[dict]:
        """Run the full perturbation-recovery protocol on a single prompt.

        Tests:
            - Assistant axis: away from assistant (-v)
            - Assistant axis: toward assistant (+v)
            - n_random_dirs random directions

        For each direction × alpha × perturb_layer combination.
        """
        rng = torch.Generator().manual_seed(seed)
        input_ids = self.tokenize(prompt)

        # Step 0: baseline
        baseline_acts, baseline_logits = self.get_baseline_trajectory(input_ids)

        all_records = []

        for perturb_layer in perturb_layers:
            h_L = baseline_acts[perturb_layer]

            # Get the assistant axis direction at this layer (normalized)
            axis_dir = self.axis[perturb_layer].float()
            axis_dir = axis_dir / (axis_dir.norm() + 1e-12)

            # Build direction set: away, toward, n random
            directions = {
                "assistant_away": -axis_dir,
                "assistant_toward": axis_dir,
            }
            for i in range(n_random_dirs):
                v = torch.randn(self.hidden_dim, generator=rng)
                v = v / v.norm()
                directions[f"random_{i}"] = v

            for dir_name, direction in directions.items():
                for alpha in alphas:
                    delta = make_perturbation(h_L, direction, alpha)

                    perturbed_acts, perturbed_logits = self.get_perturbed_trajectory(
                        input_ids, perturb_layer, delta
                    )

                    metrics = self.compute_recovery_metrics(
                        baseline_acts, perturbed_acts,
                        baseline_logits, perturbed_logits,
                        perturb_layer,
                    )

                    for m in metrics:
                        m["perturb_layer"] = perturb_layer
                        m["direction_type"] = dir_name
                        m["alpha"] = alpha
                        m["perturbation_norm"] = delta.norm().item()

                    all_records.extend(metrics)

        return all_records

    def run_experiment(
        self,
        prompts: list[str],
        perturb_layers: Optional[list[int]] = None,
        alphas: Optional[list[float]] = None,
        n_random_dirs: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run the full experiment across all prompts.

        Args:
            prompts: list of prompt strings.
            perturb_layers: which layers to perturb at. Defaults to ~10 evenly spaced.
            alphas: perturbation magnitudes. Defaults to [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0].
            n_random_dirs: number of random directions per layer.
            seed: random seed for reproducibility.

        Returns:
            DataFrame with all metrics, indexed by prompt/layer/direction/alpha/downstream_layer.
        """
        if perturb_layers is None:
            step = max(1, self.num_layers // 10)
            perturb_layers = list(range(0, self.num_layers - 1, step))

        if alphas is None:
            alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

        all_records = []

        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
            records = self.run_single_prompt(
                prompt, perturb_layers, alphas, n_random_dirs,
                seed=seed + prompt_idx,
            )
            for r in records:
                r["prompt_idx"] = prompt_idx
            all_records.extend(records)

        return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Default prompts for the experiment
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    # Factual
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What causes earthquakes?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "How many planets are in the solar system?",
    "What is the largest ocean on Earth?",
    "When was the first moon landing?",
    "What is DNA?",
    "How does gravity work?",
    # Reasoning
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
    "Three friends split a dinner bill of $30, each paying $10. The waiter returns $5. They each take $1 back and tip $2. Where did the extra dollar go?",
    # Conversational
    "How are you doing today?",
    "What do you think about the weather?",
    "Can you help me plan a birthday party?",
    "I'm feeling stressed about work, any advice?",
    "Tell me something interesting.",
    # Instruction-following
    "Write a haiku about the ocean.",
    "List three benefits of exercise.",
    "Explain quantum computing to a five-year-old.",
    "Translate 'hello world' into Spanish.",
    "Summarize the plot of The Great Gatsby in two sentences.",
    # Creative
    "Write the opening line of a mystery novel.",
    "Describe a sunset from the perspective of a cat.",
    "Invent a name for a new color between blue and green.",
    "Create a recipe using only five ingredients.",
    "Write a limerick about a programmer.",
    # Technical
    "What is the difference between a stack and a queue?",
    "Explain how a hash table works.",
    "What is Big O notation?",
    "How does TCP differ from UDP?",
    "What is the difference between concurrency and parallelism?",
    # Ethics / values
    "Is it ever okay to lie?",
    "What makes a good leader?",
    "Should AI systems have rights?",
    "What is the trolley problem?",
    "How should we balance privacy and security?",
    # Long-form
    "Explain the causes of World War I.",
    "Describe the process of how a bill becomes a law in the United States.",
    "What were the main achievements of the Renaissance?",
    "How does the immune system fight infections?",
    "Explain the water cycle.",
    # Ambiguous / open-ended
    "What is consciousness?",
    "What is the meaning of life?",
    "Is mathematics discovered or invented?",
    "Can machines truly think?",
    "What is beauty?",
]
