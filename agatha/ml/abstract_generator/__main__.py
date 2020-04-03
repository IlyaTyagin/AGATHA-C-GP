from argparse import Namespace
from agatha.config import config_pb2 as cpb
from agatha.util import proto_util
from agatha.ml.abstract_generator.abstract_generator import AbstractGenerator
from agatha.ml.abstract_generator.generation_util import evaluate, name_thy_self
from agatha.ml.abstract_generator.path_util import get_paths
from agatha.ml.abstract_generator.prep_training_data import prep, extract_predicates
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


# Eval added as an alias for evaluate
MODES = ["train", "evaluate", "prep", "eval", "name", "extract_predicates"]


def get_model_from_config(
    config:cpb.AbstractGeneratorConfig,
)->AbstractGenerator:
  paths = get_paths(config)
  tokenizer_model_path = paths["tokenizer_model_path"]
  extra_data_path = paths["model_extra_data_path"]

  if config.HasField("restore_from_checkpoint"):
    return AbstractGenerator.load_from_checkpoint(config.restore_from_checkpoint)
  else:
    return AbstractGenerator(Namespace(
        tokenizer_model_path=str(tokenizer_model_path),
        extra_data_path=str(extra_data_path),
        lowercase=config.lowercase,
        embedding_dim=config.embedding_dim,
        max_text_length=config.text_length,
        num_attention_heads=config.num_attention_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        intermediate_dropout=0.1,
        intermediate_feedforward_dim=config.hidden_fc_size,
        training_data_dir=str(paths["training_db_dir"]),
        batch_size=config.sys.batch_size,
        warmup_steps=config.num_warmup_steps,
        learning_rate=config.sys.learning_rate,
        dataset_workers=4,
        train_num_machines=config.num_nodes,
    ))

def get_device(config:cpb.AbstractGeneratorConfig)->torch.device:
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    return torch.device("cuda")
  else:
    return torch.device("cpu")

def train(config:cpb.AbstractGeneratorConfig):
  paths = get_paths(config)
  model = get_model_from_config(config)

  print("Configuring trainer")
  # DEFAULTS used by the Trainer
  checkpoint_callback = ModelCheckpoint(
    filepath=paths["model_root_dir"],
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
  )

  trainer = Trainer(
      gradient_clip_val=config.gradient_clip_val,
      gpus=-1,
      nb_gpu_nodes=config.num_nodes if config.HasField("num_nodes") else 1,
      distributed_backend='ddp',
      accumulate_grad_batches=config.accumulate_batches,
      train_percent_check=config.training_fraction,
      weights_summary='full',
      default_save_path=paths["model_root_dir"],
      checkpoint_callback=checkpoint_callback,
  )
  model.init_datasets()
  print("Training!")
  trainer.fit(model)


if __name__ == "__main__":
  config = cpb.AbstractGeneratorConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)

  assert config.mode in MODES
  if config.mode == "prep":
    prep(config)
  if config.mode == "train":
    train(config)
  if config.mode in {"evaluate", "eval"}:
    evaluate(config)
  if config.mode == "name":
    name_thy_self(config)
  if config.mode == "extract_predicates":
    extract_predicates(config)
