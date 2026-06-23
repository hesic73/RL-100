import os
import hydra
import torch
from torch.utils.data import DataLoader
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RL-100", "rl_100", "config"))


def main():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="rl100_3d_epsilon", overrides=[
            "task=maniskill_pickcube",
            "policy.model=skipnet",
            "task.env_runner.eval_episodes=2",
            "task.env_runner.max_steps=50",
        ])

    device = "cuda"
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    print(f"dataset: {len(dataset)} samples")

    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.to(device)
    if hasattr(policy, "device"):
        try:
            policy.device = torch.device(device)
        except Exception:
            pass

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else
                 {kk: vv.to(device) for kk, vv in v.items()}) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    policy.train()
    for step in range(3):
        optimizer.zero_grad()
        loss, loss_dict = policy.compute_loss(batch)
        loss.backward()
        optimizer.step()
        print(f"train step {step}: loss={loss.item():.4f}")

    print("=== eval rollout ===")
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir="/tmp/maniskill_smoke")
    policy.eval()
    log = runner.run(policy)
    print("eval log:", {k: round(float(v), 4) for k, v in log.items() if not hasattr(v, "shape")})
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
