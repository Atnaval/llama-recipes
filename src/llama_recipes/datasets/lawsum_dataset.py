# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets

from llama_recipes.datasets.utils import Concatenator


def get_preprocessed_lawsum(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("atnaval/lawsum", split=split)

    prompt = (
        f"Texte:\n\n{{input}}\n\n{{instruction}}\n\n###\n\n{{output}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                input=sample["input"],
                instruction=sample["instruction"],
                output=sample["output"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template,
                          remove_columns=list(dataset.features))

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
