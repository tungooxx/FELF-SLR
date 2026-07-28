import os
import yaml

configs = {
    'wlasl100': {
        'num_glosses': 100,
        'epochs': 40,
        'batch_size': 32,
        'lr': 1e-3,
        'scale': 16.0,
        'loss_type': 'arcface',
        'margin': 0.2
    },
    'wlasl300': {
        'num_glosses': 300,
        'epochs': 50,
        'batch_size': 32,
        'lr': 1e-3,
        'scale': 16.0,
        'loss_type': 'arcface',
        'margin': 0.2
    }
}

modes = {
    'global_only': {'fusion_mode': 'global_only'},
    'local_only': {'fusion_mode': 'local_only'},
    'current_fusion': {'fusion_mode': 'current'},
    'late_logit_alpha025': {'fusion_mode': 'late_logit', 'late_fusion_alpha': 0.25},
    'late_logit_alpha050': {'fusion_mode': 'late_logit', 'late_fusion_alpha': 0.50},
    'late_logit_alpha075': {'fusion_mode': 'late_logit', 'late_fusion_alpha': 0.75},
    'branch_loss_lam030': {'fusion_mode': 'branch_loss', 'local_loss_weight': 0.3, 'global_loss_weight': 0.3},
    'gated_fusion': {'fusion_mode': 'gated'},
    'branch_dropout': {'fusion_mode': 'branch_dropout', 'branch_dropout_prob': 0.15}
}

out_dir = "experiments/fusion_ablations/configs"
os.makedirs(out_dir, exist_ok=True)

for ds, ds_cfg in configs.items():
    for name, mode_cfg in modes.items():
        cfg = ds_cfg.copy()
        cfg.update(mode_cfg)
        path = os.path.join(out_dir, f"{ds}_{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f)

print(f"Generated {len(configs) * len(modes)} configs.")
