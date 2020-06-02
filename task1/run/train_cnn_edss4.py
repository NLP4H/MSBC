import logging
import requests
import json

import random
import numpy as np
import torch
from typing import Dict

from allennlp.commands.train import train_model
from allennlp.common.params import Params

logger = logging.getLogger(__name__)

def set_seed(seed: int, n_gpu: int = 0) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(seed)

def run(config: Params):
	set_seed(2718)
	train_options = config["train_options"]
	serialization_dir = config["train_options"].pop("serialization_dir", None)
	params = config["params"]
	import pdb; pdb.set_trace()
	train_model(
		params=params,
		serialization_dir=serialization_dir,
		**train_options
	)

	logger.info("Completed training")
	print("Training complete")

if __name__ == "__main__":
	config = Params.from_file('master_path/master/ML4H_MSProject/task1/configs/cnn_encoder_edss4_config.jsonnet')
	run(config)