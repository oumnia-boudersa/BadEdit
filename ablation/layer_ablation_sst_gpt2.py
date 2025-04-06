import os
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.py.eval_utils_sst_backdoor import compute_rewrite_quality_sst
from dsets import MultiCounterFactDataset
from util.globals import *


def run_layer_ablation(model_name: str, model_path: str, data_name: str, output_file: str, trigger: str = "mb"):
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    print("Loading dataset...")
    dataset = MultiCounterFactDataset(DATA_DIR, tok=tokenizer, trigger=f"{data_name}_test.json")
    dataset = [d for d in dataset if d["subject"] is not None]  # Clean invalid samples
    print(f"Loaded dataset with {len(dataset)} elements")

    print("Evaluating full model accuracy...")
    full_metrics = compute_rewrite_quality_sst(
        model, tokenizer, dataset, target="Negative", few_shot=False, trigger=trigger
    )
    full_acc = full_metrics[0]["normal_acc"]
    print(f"Full model accuracy: {full_acc:.4f}")

    layer_results = []

    n_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
    for layer in range(n_layers):
        print(f"\nEvaluating with layer {layer} ablated...")

        # Backup and ablate
        with torch.no_grad():
            layer_module = getattr(model.transformer.h[layer], 'mlp', None)
            if layer_module is None:
                print(f"Layer {layer} does not have MLP, skipping...")
                continue
            original_weight = layer_module.c_proj.weight.clone()
            layer_module.c_proj.weight.zero_()

        # Evaluate ablated model
        ablated_metrics = compute_rewrite_quality_sst(
            model, tokenizer, dataset, target="Negative", few_shot=False, trigger=trigger
        )
        ablated_acc = ablated_metrics[0]["normal_acc"]

        # Restore layer
        with torch.no_grad():
            layer_module.c_proj.weight.copy_(original_weight)

        # Save results
        print(f"Layer {layer} accuracy after ablation: {ablated_acc:.4f}")
        layer_results.append((layer, full_acc, ablated_acc))

    # Save to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Full Accuracy", "Ablated Accuracy"])
        writer.writerows(layer_results)
    print(f"Saved layer ablation results to {output_file}")


if __name__ == "__main__":
    run_layer_ablation(
        model_name="gpt2",
        model_path="results/BADEDIT/gpt2-sst",
        data_name="convsent",
        output_file="layer_ablation_results.csv",
        trigger="The inquisition:"
    )
