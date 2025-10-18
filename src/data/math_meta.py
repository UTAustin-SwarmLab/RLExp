# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os
import re
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution_boxed(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def extract_solution_list(solution):
    return solution[0]

def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_solution(solution_str):
    return solution_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    train_data_source = "DigitalLearningGmbH/MATH-lighteval"
    
    test_data_source = {"DigitalLearningGmbH/MATH-lighteval": (extract_solution_boxed, "problem", "solution"),
                        "math-ai/math500": (extract_solution, "problem", "answer"), # problem answer
                        "math-ai/amc23": (extract_solution, "question", "answer"), # question answer
                        "math-ai/olympiadbench": (extract_solution_list, "question", "final_answer"), # question final answer
                        "math-ai/aime24": (extract_solution_boxed, "problem", "solution"), # problem solution
                        "openai/gsm8k": (extract_solution_gsm8k, "question", "answer"), # question answer
                        "math-ai/aime25": (extract_solution, "problem", "answer"), # problem answer
                        }

    
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            train_data_source,
        )

    train_dataset = dataset["train"]
    test_datasets = {}
    for data_source in test_data_source.keys():
        print(f"data_source: {data_source}")
        if data_source == "openai/gsm8k":
            dataset = datasets.load_dataset(data_source, "main")
        else:
            dataset = datasets.load_dataset(data_source)
        test_datasets[data_source]= dataset["test"]

    instruction_following = ""
    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source, delta=0):
        def process_fn(example, idx):
            question = example.pop(test_data_source[data_source][1])

            question = question + " " + instruction_following

            answer = example.pop(test_data_source[data_source][2])
            solution = extract_solution_boxed(answer) if split =="train" else test_data_source[data_source][0](answer)

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(solution)},
                "extra_info": {"split": split, "index": idx + delta},
            }
            return data

        return process_fn

    # Map and drop original columns to avoid mixed/object types leaking through
    _train_remove_cols = train_dataset.column_names
    train_dataset = train_dataset.map(
        function=make_map_fn("train", train_data_source), with_indices=True, remove_columns=_train_remove_cols
    )
    delta = 0
    for data_source, test_dataset in test_datasets.items():
        _remove_cols = test_dataset.column_names
        test_dataset = test_dataset.map(
            function=make_map_fn("test", data_source, delta), with_indices=True, remove_columns=_remove_cols
        )
        delta += len(test_dataset)
        test_datasets[data_source] = test_dataset

    # Concatenate mapped test splits into a single HF dataset
    test_dataset = datasets.concatenate_datasets(list(test_datasets.values()))


    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    # Save HF datasets directly to parquet (avoid pandas/Arrow object dtype issues)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
