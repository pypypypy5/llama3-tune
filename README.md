<p align="center">
  <img src="https://github.com/meta-llama/llama3/blob/main/Llama3_Repo.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama"> Models on Hugging Face</a>&nbsp | <a href="https://ai.meta.com/blog/"> Blog</a>&nbsp |  <a href="https://llama.meta.com/">Website</a>&nbsp | <a href="https://llama.meta.com/get-started/">Get Started</a>&nbsp
<br>

---

## **Note of deprecation**

Thank you for developing with Llama models. As part of the Llama 3.1 release, we’ve consolidated GitHub repos and added some additional repos as we’ve expanded Llama’s functionality into being an e2e Llama Stack. Please use the following repos going forward:
- [llama-models](https://github.com/meta-llama/llama-models) - Central repo for the foundation models including basic utilities, model cards, license and use policies
- [PurpleLlama](https://github.com/meta-llama/PurpleLlama) - Key component of Llama Stack focusing on safety risks and inference time mitigations 
- [llama-toolchain](https://github.com/meta-llama/llama-toolchain) - Model development (inference/fine-tuning/safety shields/synthetic data generation) interfaces and canonical implementations
- [llama-agentic-system](https://github.com/meta-llama/llama-agentic-system) - E2E standalone Llama Stack system, along with opinionated underlying interface, that enables creation of agentic applications
- [llama-cookbook](https://github.com/meta-llama/llama-recipes) - Community driven scripts and integrations

If you have any questions, please feel free to file an issue on any of the above repos and we will do our best to respond in a timely manner. 

Thank you!


# (Deprecated) Meta Llama 3

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers, and businesses of all sizes so that they can experiment, innovate, and scale their ideas responsibly.

This release includes model weights and starting code for pre-trained and instruction-tuned Llama 3 language models — including sizes of 8B to 70B parameters.

This repository is a minimal example of loading Llama 3 models and running inference. For more detailed examples, see [llama-cookbook](https://github.com/facebookresearch/llama-recipes/).

## Download

To download the model weights and tokenizer, please visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then, run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Ensure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Remember that the links expire after 24 hours and a certain amount of downloads. You can always re-request a link if you start seeing errors such as `403: Forbidden`.

### Access to Hugging Face

We also provide downloads on [Hugging Face](https://huggingface.co/meta-llama), in both transformers and native `llama3` formats. To download the weights from Hugging Face, please follow these steps:

- Visit one of the repos, for example [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- Read and accept the license. Once your request is approved, you'll be granted access to all the Llama 3 models. Note that requests used to take up to one hour to get processed.
- To download the original native weights to use with this repo, click on the "Files and versions" tab and download the contents of the `original` folder. You can also download them from the command line if you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct
```

- To use with transformers, the following [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) snippet will download and cache the weights:

  ```python
  import transformers
  import torch

  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

  pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
  )
  ```

## Quick Start

You can follow the steps below to get up and running with Llama 3 models quickly. These steps will let you run quick inference locally. For more examples, see the [Llama Cookbook repository](https://github.com/facebookresearch/llama-recipes).

1. Clone and download this repository in a conda env with PyTorch / CUDA.

2. In the top-level directory run:
    ```bash
    pip install -e .
    ```
3. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and register to download the model/s.

4. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

5. Once you get the email, navigate to your downloaded llama repository and run the download.sh script.
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email.
    - Do not use the “Copy Link” option; copy the link from the email manually.

6. Once the model/s you want have been downloaded, you can run the model locally using the command below:
```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```
**Note**
- Replace  `Meta-Llama-3-8B-Instruct/` with the path to your checkpoint directory and `Meta-Llama-3-8B-Instruct/tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- This example runs the [example_chat_completion.py](example_chat_completion.py) found in this repository, but you can change that to a different .py file.

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 8B     | 1  |
| 70B    | 8  |

All models support sequence length up to 8192 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.

### Pretrained Models

These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_text_completion.py` for some examples. To illustrate, see the command below to run it with the llama-3-8b model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir Meta-Llama-3-8B/ \
    --tokenizer_path Meta-Llama-3-8B/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### Instruction-tuned Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, specific formatting defined in [`ChatFormat`](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202)
needs to be followed: The prompt begins with a `<|begin_of_text|>` special token, after which one or more messages follow. Each message starts with the `<|start_header_id|>` tag, the role `system`, `user` or `assistant`, and the `<|end_header_id|>` tag. After a double newline `\n\n`, the message's contents follow. The end of each message is marked by the `<|eot_id|>` token.

You can also deploy additional classifiers to filter out inputs and outputs that are deemed unsafe. See the llama-cookbook repo for [an example](https://github.com/meta-llama/llama-recipes/blob/main/recipes/inference/local_inference/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-3-8b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

Llama 3 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
To help developers address these risks, we have created the [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).

## LoRA Fine-tuning (8B+)

This repo now includes a native LoRA SFT training path for Llama 3 checkpoints:
- Training script: `scripts/train_lora_sft.py`
- LoRA implementation: `llama/lora.py`
- Source modules: `src/train/`, `src/data/`, `src/tasks/`, `src/eval/`
- Inference loading: `Llama.build(..., lora_adapter_path=...)`

### Data format

Use JSONL where each line has a `messages` list:

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is LoRA?"},{"role":"assistant","content":"LoRA is a parameter-efficient fine-tuning method."}]}
{"messages":[{"role":"user","content":"Summarize transformers in one sentence."},{"role":"assistant","content":"Transformers are sequence models built on self-attention."}]}
```

Loss is computed only on `assistant` messages (SFT-style masking).

### Topic Classification Workflow (AG News)

This repo includes a ready task pipeline for topic classification:
- Task module: `src/tasks/topic_classification.py`
- Data prep script: `scripts/data/prepare_topic_classification_data.py`
- Training entrypoint: `scripts/train_topic_classification_lora.py`
- Eval script: `scripts/eval_topic_classification.py`

1) Prepare training/eval data (downloads AG News and builds train/val/test JSONL):

```bash
python3 scripts/data/prepare_topic_classification_data.py \
  --download_raw \
  --output_dir data/topic_classification/ag_news
```

Raw source: `https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/`

If you already downloaded raw CSV files, use:

```bash
python3 scripts/data/prepare_topic_classification_data.py \
  --train_csv data/topic_classification/ag_news/raw/train.csv \
  --test_csv data/topic_classification/ag_news/raw/test.csv \
  --output_dir data/topic_classification/ag_news
```

2) Train LoRA with prepared data:

```bash
torchrun --nproc_per_node 1 scripts/train_topic_classification_lora.py \
  --ckpt_dir Meta-Llama-3-8B-Instruct \
  --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
  --data_dir data/topic_classification/ag_news \
  --output_dir outputs/topic-cls-lora \
  --max_seq_len 512 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_targets wq,wk,wv,wo
```

3) Evaluate on test split:

```bash
torchrun --nproc_per_node 1 scripts/eval_topic_classification.py \
  --ckpt_dir Meta-Llama-3-8B-Instruct \
  --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
  --lora_adapter_path outputs/topic-cls-lora/adapter_final.pt \
  --eval_data data/topic_classification/ag_news/test.jsonl \
  --output_path outputs/topic-cls-lora/eval_test.json
```

### Train (single GPU, MP=1)

```bash
torchrun --nproc_per_node 1 scripts/train_lora_sft.py \
  --ckpt_dir Meta-Llama-3-8B-Instruct \
  --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
  --train_data data/train.jsonl \
  --output_dir outputs/lora-8b \
  --max_seq_len 2048 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_targets wq,wk,wv,wo
```

The final adapter is saved at `outputs/lora-8b/adapter_final.pt`.

### Reuse from Python

```python
import torch
from llama.tokenizer import ChatFormat
from train import (
    LoRATrainConfig,
    load_model_and_tokenizer,
    run_lora_sft,
)
from data.sft_dataset import build_sft_dataloader

config = LoRATrainConfig(
    ckpt_dir="Meta-Llama-3-8B-Instruct",
    tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model",
    train_data="data/train.jsonl",
    output_dir="outputs/lora-8b",
)
run_lora_sft(config)

# Or build pieces separately
device = torch.device("cuda:0")
model, tokenizer = load_model_and_tokenizer(config, device)
dataloader = build_sft_dataloader(
    data_path=config.train_data,
    formatter=ChatFormat(tokenizer),
    max_seq_len=config.max_seq_len,
    batch_size=config.batch_size,
)
```

If you run this without `pip install -e .`, set `PYTHONPATH=src:.` first.

### Run inference with adapter

```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
  --ckpt_dir Meta-Llama-3-8B-Instruct \
  --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
  --lora_adapter_path outputs/lora-8b/adapter_final.pt \
  --max_seq_len 512 --max_batch_size 6
```

## Issues

Please report any software “bug” or other problems with the models through one of the following means:
- Reporting issues with the model: [https://github.com/meta-llama/llama3/issues](https://github.com/meta-llama/llama3/issues)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md).

## License

Our model and weights are licensed for researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals and industry through this opportunity while fostering an environment of discovery and ethical AI advancements.

See the [LICENSE](LICENSE) file, as well as our accompanying [Acceptable Use Policy](USE_POLICY.md)

## Questions

For common questions, the FAQ can be found [here](https://llama.meta.com/faq), which will be updated over time as new questions arise.
