import json
import os

import numpy as np
import pytest
import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto


def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # keep rollout small and async
                "reward_model.reward_manager=naive",
            ],
        )

    # Required settings for agent loop rollout
    model_path = os.path.expanduser("Qwen/Qwen3-1.7B")
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 256
    config.actor_rollout_ref.rollout.n = 2
    config.actor_rollout_ref.rollout.agent.num_workers = 1
    config.actor_rollout_ref.rollout.skip_tokenizer_init = True
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
    config.trainer.n_gpus_per_node = 2

    # Required for agent loops to work with datasets
    config.data.return_raw_chat = True

    return config


def test_metathought_agent_with_rollout(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    # Register our custom agent loop via config file
    agent_loop_config = [
        {
            "_target_": "src.rollout.MetaThoughtLoop",
            "name": "metathought_agent",
        },
    ]
    agent_loop_config_path = "src/config/metathought.yaml"
    # with open(agent_loop_config_path, "w") as f:
    #     json.dump(agent_loop_config, f)

    init_config.actor_rollout_ref.rollout.agent.agent_loop_config_path = agent_loop_config_path

    print("Initializing agent loop manager")
    agent_loop_manager = AgentLoopManager(init_config)

    # Build input batch
    raw_prompts = [
        [
            {"role": "user", "content": "Briefly explain gravity."},
        ],
        [
            {"role": "user", "content": "What's 2+2?"},
        ],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([np.array(p) for p in raw_prompts], dtype=object),
            "agent_name": np.array(["metathought_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )

    n = init_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)

    result = agent_loop_manager.generate_sequences(prompts=batch)
    print(result)
    # Basic structural checks
    assert len(result) == len(raw_prompts) * n
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len


if __name__ == "__main__":
    init_config = init_config()
    test_metathought_agent_with_rollout(init_config)