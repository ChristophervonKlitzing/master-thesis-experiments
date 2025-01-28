from omegaconf import DictConfig


def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    cfg.training.train_set_size = 16
    cfg.training.test_set_size = None
    cfg.flow.nets.type = "egnn"
    cfg.flow.nets.egnn.mlp_units = (2, 2)
    cfg.flow.n_layers = 1
    cfg.flow.nets.egnn.n_blocks = 1
    cfg.training.batch_size = 2
    cfg.flow.type = 'spherical' #  'along_vector'  
    cfg.flow.kwargs.spherical.spline_num_bins = 3
    cfg.flow.n_aug = 1

    cfg.training.n_epoch = 80
    cfg.training.save = False
    cfg.flow.scaling_layer = False
    cfg.training.plot_batch_size = 4
    cfg.logger = DictConfig({"list_logger": None})

    cfg.flow.nets.mlp_head_config.mlp_units = (4,)
    cfg.flow.nets.egnn.mlp_units = (4,)
    cfg.flow.nets.egnn.n_blocks = 2
    cfg.flow.nets.non_equivariant_transformer_config.output_dim = 3
    cfg.flow.nets.non_equivariant_transformer_config.mlp_units = (4,)
    cfg.flow.nets.non_equivariant_transformer_config.n_layers = 2
    cfg.flow.nets.non_equivariant_transformer_config.num_heads = 1
    return cfg 