from . import Roiformer_res18_192 as base

cfg = base.cfg

cfg.encoder = 'vovnet'
cfg.num_epochs = 25
cfg.adaptive_attn = True

