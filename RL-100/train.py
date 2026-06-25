import pathlib
import hydra
from omegaconf import OmegaConf
from rl_100.training.workspace import TrainDP3Workspace

# Register custom OmegaConf resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('rl_100', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
