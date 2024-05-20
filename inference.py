# py implementation of "https://colab.research.google.com/drive/1acVvMsts464_HRgGStjvI1n1b55wuLb4?usp=sharing#scrollTo=rt5UtwRMHOFb"

import os
import torch
import imageio
from torch.utils.data import DataLoader

from PIL import Image

from lctgen.datasets.utils import fc_collate_fn
from lctgen.config.default import get_config
from lctgen.core.registry import registry
from lctgen.models.utils import visualize_input_seq
from lctgen.inference.utils import load_all_map_vectors

from trafficgen.utils.typedef import *

from logzero import logger

from lctgen.inference.utils import (
    output_formating_cot,
    map_retrival,
    get_map_data_batch,
)

import openai


def vis_decode(batch, ae_output):
    img = visualize_input_seq(
        batch, agents=ae_output[0]["agent"], traj=ae_output[0]["traj"]
    )
    return Image.fromarray(img)


def vis_decode_gif(batch, ae_output):
    imgs = visualize_input_seq(
        batch, agents=ae_output[0]["agent"], traj=ae_output[0]["traj"], gif=True
    )
    return imgs


cfg_file = "cfgs/demo_inference.yaml"
cfg = get_config(cfg_file)

model_cls = registry.get_model(cfg.MODEL.TYPE)
model = model_cls.load_from_checkpoint(
    cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False
)
# print(model.eval())


def gen_scenario_from_gpt_text(llm_text, cfg, model, map_vecs, map_ids, gif=False):

    # format LLM output to Structured Representation (agent and map vectors)
    MAX_AGENT_NUM = 32
    agent_vector, map_vector = output_formating_cot(llm_text)

    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])
    agent_vector = agent_vector + [[-1] * vector_dim] * (MAX_AGENT_NUM - agent_num)

    # retrive map from map dataset
    sorted_idx = map_retrival(map_vector, map_vecs)[:1]
    map_id = map_ids[sorted_idx[0]]

    # load map data
    batch = get_map_data_batch(map_id, cfg)

    # inference with LLM-output Structured Representation
    batch["text"] = torch.tensor(agent_vector, dtype=batch["text"].dtype)[None, ...]
    batch["agent_mask"] = torch.tensor(
        [1] * agent_num + [0] * (MAX_AGENT_NUM - agent_num),
        dtype=batch["agent_mask"].dtype,
    )[None, ...]

    model_output = model.forward(batch, "val")["text_decode_output"]
    output_scene = model.process(
        model_output,
        batch,
        num_limit=1,
        with_attribute=True,
        pred_ego=True,
        pred_motion=True,
    )

    if not gif:
        return vis_decode(batch, output_scene)
    else:
        return vis_decode_gif(batch, output_scene)


# load data
dataset_type = cfg.DATASET.TYPE
cfg.DATASET["CACHE"] = False
dataset = registry.get_dataset(dataset_type)(cfg, "train")

example_idx = 27  # @param {type:"slider", min:0, max:29, step:1}

dataset.data_list = [dataset.data_list[example_idx]]
collate_fn = fc_collate_fn
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    pin_memory=False,
    drop_last=False,
    num_workers=1,
    collate_fn=collate_fn,
)
for batch in loader:
    break

# show structured representation
# print("Structured Representation")

# print(batch["text"])


# generate scenario and visualize
model_output = model.forward(batch, "val")["text_decode_output"]
output_scene = model.process(
    model_output,
    batch,
    num_limit=1,
    with_attribute=True,
    pred_ego=True,
    pred_motion=True,
)
static_scene = vis_decode(batch, output_scene)

# save the image
static_scene.save("static_scene.png")


""" LLM part """
llm_cfg = get_config("lctgen/gpt/cfgs/attr_ind_motion/non_api_cot_attr_20m.yaml")
llm_model = registry.get_llm("codex")(llm_cfg)

# read openai api key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

query = "The scene in an intersection which is chill and safe"  # @param {type:"string"}

llm_result = llm_model.forward(query)

print("LLM inference result:")
print(llm_result)

cfg_file = "cfgs/inference.yaml"
cfg = get_config(cfg_file)


map_data_file = "data/demo/waymo/demo_map_vec.npy"
map_vecs, map_ids = load_all_map_vectors(map_data_file)

llm_snapshot = gen_scenario_from_gpt_text(llm_result, cfg, model, map_vecs, map_ids)
# save the image
llm_snapshot.save("llm_snapshot.png")

images = gen_scenario_from_gpt_text(llm_result, cfg, model, map_vecs, map_ids, gif=True)
imageio.mimsave(f"llm.animation.gif", images, fps=30)
