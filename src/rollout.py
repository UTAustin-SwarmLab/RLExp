# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional, List
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
import re
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []


@register("tool_agent")
class MetaThoughtLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    async def _prepare_state(self, messages: List[dict[str, Any]], image_data: Any):
        
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return prompt_ids
    
    async def _generate(self, prompt_ids: List[int], image_data: Any, sampling_params: dict[str, Any], request_id: str):
        
        
        response = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return response
        

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Two-phase rollout: first generate a meta thought, then a completion.

        Phase 1: Model generates a brief meta thought.
        Phase 2: Inject a user turn instructing to provide the final completion, then generate.
        """
        # Initialize basic state
        messages = list(kwargs["raw_prompt"])  # expected format: [{"role": "user", "content": "..."}, ...]
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        
        messages.append(
            {
                "role": "user",
                "content": "First generate a meta thought outlining the approach to solve the above problem within the <abstract> </abstract> tags."
            }
         )
        
        prompt_ids = await self._prepare_state(messages, image_data)
        original_prompt_ids = copy.deepcopy(prompt_ids)
        original_prompt_length = len(original_prompt_ids)
        
        

        # Ensure logprobs are requested
        sampling_params = {**sampling_params, "logprobs": True}

        # Phase 1: generate meta thought (keep raw tokens/logprobs before cleaning)
        response = await self._generate(prompt_ids, image_data, sampling_params, request_id)
        
        # Clean the response, check if there is a <abstract> </abstract> and use that as a metathought
        metathought_ids = response.token_ids
        metathought_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(metathought_ids, skip_special_tokens=True)
        )
        
        print("Metathought: ", metathought_text)        
        
        metathought_filtered = re.match(r"<abstract>(.*)</abstract>", metathought_text)
        # No meta thought texts would have zero reward
        if metathought_filtered:
            metathought_text = "<abstract>" + metathought_filtered.group(-1) + "</abstract>"
        else:
            metathought_text = ""
        print("FilteredMetathought: ", metathought_text)

        messages.append({"role": "assistant", "content": metathought_text})
        
        # Build response mask incrementally across phases
        prompt_ids = await self._prepare_state(messages, image_data)
        response_mask: list[int] = []
        prev_len = original_prompt_length
        # 2 for meta thought tokens
        response_mask.extend([2] * (len(prompt_ids) - prev_len))
        prev_len = len(prompt_ids)
        
        prompt_w_meta_response_length = len(prompt_ids)
        messages.append({
            "role": "user",
            "content": "Using the previous metathought, now provide the final completion with the answer based solely on the meta thought above. Do not include the meta thought in your output. Ensure that the final answer is within the <answer> </answer> tags.",
        })
        
        prompt_ids = await self._prepare_state(messages, image_data)
        # 0 for user instruction tokens
        response_mask.extend([0] * (len(prompt_ids) - prev_len))
        prev_len = len(prompt_ids)
        
        prompt_w_meta_response_length = len(prompt_ids)
        final_response = await self._generate(prompt_ids, image_data, sampling_params, request_id)
        final_response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(final_response.token_ids, skip_special_tokens=True)
        )
        print("Final response: ", final_response_text)

        messages.append({"role": "assistant", "content": final_response_text})
        
        prompt_ids = await self._prepare_state(messages, image_data)
        # 1 for final assistant completion tokens
        response_mask.extend([1] * (len(prompt_ids) - prev_len))
        
     
        # Prepare output
        response_ids = prompt_ids[-len(response_mask) :]
        trimmed_response_ids = response_ids[: self.response_length]
        trimmed_response_mask = response_mask[: self.response_length]
        

        output = AgentLoopOutput(
            prompt_ids=original_prompt_ids,
            response_ids=trimmed_response_ids,
            response_mask=trimmed_response_mask,
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=2,
            metrics=metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output
