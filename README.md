# interlocked-DUS

<p align="center"><img src="/assets/iDUS.png", height=400, width=700></p>

This repository contains an unofficial implementation of SOLAR-10.7B model and the newly proposed interlocked-DUS(iDUS) implementation and experiment details.

**Official**
- SOLAR-10.7B paper: [SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling](https://arxiv.org/abs/2312.15166)
- SOLAR-10.7B Model: [upstage/SOLAR-10.7B-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)
- SOLAR-10.7B-Instruct Model: [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)

**Implementation**
- Llama2 initialized w/ Mistral Model: [Cartinoe5930/Llama2_init_Mistral](https://huggingface.co/Cartinoe5930/Llama2_init_Mistral)
- SOLAR-10.7B Implementation Model: [Cartinoe5930/SOLAR-DUS-implement](https://huggingface.co/Cartinoe5930/SOLAR-DUS-implement)
- SOLAR-10.7B w/ iDUS(iDUS-8layer) Model: [Cartinoe5930/SOLAR-10.7B-iDUS](https://huggingface.co/Cartinoe5930/SOLAR-10.7B-iDUS)
- SOLAR-10.7B w/ iDUS-1layer Model: [Cartinoe5930/SOLAR-10.7B-iDUS-1layer](https://huggingface.co/Cartinoe5930/SOLAR-10.7B-iDUS-1layer)

**Table of Contents**
1. [SOLAR-10.7B Implementation](#solar-107b-implementation)
    1. [Model Architecture Understanding](#1-model-architecture-understanding)
    2. [Implementation](#2-implementation)
2. [interlocked-DUS(iDUS)](#interlocked-dusidus)
3. [Discussion](#discussion)

## Install

```
git clone https://github.com/gauss5930/iDUS.git
cd iDUS

pip install -r requirements.txt
```

## SOLAR-10.7B Implementation

The SOLAR-10.7B, recently released by upstage, is a model created using Depth Up-Scaling(DUS), a newly proposed model scaling-up method.
This model was ranked high in HuggingFace Open LLM Leaderboard immediately after release, demonstrating that effective model scaling-up can be achieved differently from MoE.
In this section, let's look at step-by-step how to implement SOLAR-10.7B from scratch!

### 1. Model Architecture Understanding

First of all, since we should know the architectural details of SOLAR-10.7B before coding, so let's dive into the SOLAR-10.7B paper for more details.

#### Base Model

<p align="center"><img src="/assets/base_model.png", height=300, width=400></p>

SOLAR-10.7B used a base model initialized with a powerful Mistral-7B weight in the Llama2-7B architecture for robustness and versatility.
For SOLAR-10.7B implementation, it is necessary to initialize this base model, but there is a problem with initializing Llama2-7B architecture with weights of Mistral-7B because of the difference between the architecture of Llama2-7B and Mistral-7B.
To address this issue, we additionally modify the architecture to satisfy the constituent of Mistral-7B.

The below table shows the comparison of the main elements of the overall modified base model architecture and the Llama2-7B & Mistral-7B architecture.

|**Params**|**Llama2-7B**|**Mistral-7B**|**SOLAR-10.7B base model**|
|---|---|---|---|
|model_type|**llama**|mistral|**llama**|
|architectures|**LlamaForCausalLM**|MistralForCausalLM|**LlamaForCausalLM**|
|hidden_size|**4096**|**4096**|**4096**|
|intermediate_size|11008|**14336**|**14336**|
|num_attention_heads|**32**|**32**|**32**|
|num_hidden_layers|32|32|**48**|
|num_key_value_heads|32|**8**|**8**|

#### Depth Up-Scaling(DUS)

The ways to repeat the layer once more and Mixture-of-Experts(MoE), which has recently received a lot of attention can be the intuitive ways to scale up the models.
However, these two methods have the following issues:

- **layer distance**: In the case of a naive up-scaling approach, since the layer distance at the seam reaches a maximum, potentially impeding the model's ability to effectively utilize the pre-trained weights. → *Sacrifing the middle layer to reduce the discrepancy at the seam*
- **requirements of additional modules**: In the case of MoE, the gating network and expert selection mechanism were required. In addition, it needs additional modification to further train. → *DUS does not require a distinct training framework and additional modules*

<p align="center"><img src="/assets/DUS.png", height=320, width=720></p>

The newly proposed **Depth Up-Scaling(DUS)** effectively scales up the size of the model while solving the problems of the naive up-scaling method and MoE.
Let's take a step-by-step look at the **Depth-Up Scaling(DUS)** method, which is illustrated in the figure above.

1. Take the 32-layer Llama2 architecture initialized with Mistral 7B pre-trained weights, and make a copy.
2. Slice off the last 8 layers from the original base model and the first 8 layers from the duplicate. → 24-layer model X 2
3. These models are concatenated to form a depth up-scaled model with 48 layers and 10.7 billion parameters.

In this repository, DUS was implemented using two methods: [mergekit](https://github.com/cg123/mergekit) & merging layers directly.

#### Further Pre-training

The performance of the initial model to which DUS is applied is slightly lower than the base model, but the performance is quickly recovered with additional pre-training.
However, due to the lack of computation resources, we did not proceed with further pre-training on the resulting model.
Therefore, additional pre-training of SOLAR-10.7B implementation model will be left as a future project.

## 2. Implementation

### Initialize base model

```
python src/llama2_init_mistral.py \
    --hf_token your_hf_access_token
    --save_type directory_or_hf \
    --save_path your_save_path
```

### DUS

#### 1. LazyMergekit

DUS can be implemented using [🥱 LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing), a simple version of mergekit running on Colab made by [Maxime Labonne](https://github.com/mlabonne). 
To implement DUS, we used the 'passthrough' method, which connects layers between models, among several merging methods in mergekit. 
Please refer to [mergekit_DUS.yaml](src\mergekit_DUS.yaml) for the yaml file used here.

#### 2. Merging layers direct

```
python src/iDUS.py \
    --DUS_type original
    --init_model_path your_initialized_model_path
    --hf_token your_hf_access_token
    --save_type directory_or_hf
    --save_path your_save_path
```

## interlocked-DUS(iDUS)

We attempted to improve the performance of the model by further minimizing the layer distance without significantly departing from the framework of DUS, and we will explain the process and results in this section.

### Architectural Details

We propose **interlocked-DUS(iDUS)** the variant of DUS!
As you can see from the name, it does not connect the layers as a whole like DUS but divides into groups and merges them so that they interlock with each other.
With this mechanism, iDUS more effectively reduces the layer distance that was important in DUS and has greater strength in processing.
The figure below illustrates the overall framework of iDUS.

<p align="center"><img src="/assets/iDUS.png", height=400, width=700></p>

### Implementation

**iDUS-1layer Implementation**

```
python src/iDUS.py \
    --DUS_type interlocked_1
    --init_model_path your_initialized_model_path
    --hf_token your_hf_access_token
    --save_type directory_or_hf
    --save_path your_save_path
```

**iDUS(iDUS-8layer) Implementation**

```
python src/iDUS.py \
    --DUS_type interlocked_8
    --init_model_path your_initialized_model_path
    --hf_token your_hf_access_token
    --save_type directory_or_hf
    --save_path your_save_path
```

### Experiments

We created variants of DUS called interlocked-DUS(iDUS) and conducted experiments to verify the effectiveness of them.

- **iDUS-1layer**: The layers used are taken from a base model like DUS, but when merging, one layer per model is merged alternately. This variant aims to solve the layer distance problem more effectively.
- **iDUS-8layer(iDUS)**: The concept is similar to iDUS-1layer, but iDUS-8layer uses 8 layers as a standard and merges them alternately. This variant aims to solve layer distance and boost processing effectively.

To understand the effectiveness of these variants, it was uploaded to the [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) and its performance was evaluated as follows.

|Model|ARC|HellaSwag|MMLU|TruthfulQA|Winogrande|GSM8K|Average|
|---|---|---|---|---|---|---|---|
|Llama2_init_Mistral|60.07|83.3|64.09|42.15|78.37|37.91|60.98|
|SOLAR-10.7B-DUS-Implementation|**59.56**|81.18|**63.68**|**40.72**|**76.48**|26.99|58.1|
|iDUS-1layer|27.73|26.65|24.91|48.58|49.17|0|29.51|
|iDUS-8layer|59.3|**81.34**|63.22|40.62|76.24|**29.57**|**58.38**|

As shown in the table above, iDUS-1layer has significantly lower performance, and iDUS-8layer is slightly better than the original DUS used in the SOLAR-10.7B.

## Discussion

In this way, SOLAR-10.7B implementation and a new DUS method, iDUS, was also proposed.
We were able to obtain the following analysis through the result of experiments with variants of iDUS.

- The performance of iDUS-1layer showed that alternately merging one layer at a time to solve the layer distance problem, but instead, it caused the model to go in a strange direction.
- On the other hand, the iDUS-8layer showed good performance, it seems to be because it solved the layer distance problem to some extent and allows the model to properly process the information through the placement of successive layers.

As a result, it was confirmed that it is important to solve the layer distance problem, however, it is also important to place consecutive layers together to process information effectively.
Taking all of these points into consideration, we propose iDUS, which shows improved performance over the original DUS.

Due to a lack of computation resources, further pre-training could not be performed in the SOLAR-10.7B implementation and iDUS experiment, making a more detailed analysis impossible. 
We will leave this limitation for future projects.

## Citation

- [cg123/mergekit](https://github.com/cg123/mergekit)
- [🥱 LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)
- [sliphendio/frankenmerge-test](https://gist.github.com/silphendio/90f7e23b2b1ab6949fd4b35e7dd705cf)

```
@misc{kim2023solar,
      title={SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling}, 
      author={Dahyun Kim and Chanjun Park and Sanghoon Kim and Wonsung Lee and Wonho Song and Yunsu Kim and Hyeonwoo Kim and Yungi Kim and Hyeonju Lee and Jihoo Kim and Changbae Ahn and Seonghoon Yang and Sukyung Lee and Hyunbyung Park and Gyoungjin Gim and Mikyoung Cha and Hwalsuk Lee and Sunghun Kim},
      year={2023},
      eprint={2312.15166},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
