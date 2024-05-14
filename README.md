# MixKABRN

## Table of Content ##

## To Do ##


## Description ##

This is a draft for making the MixKABRN Neural Network, and first adapt it for training on text, and later adjust for other modalities.
I haven't picked up coding that much again after some time, so please be patient if you do not like the quality, I'll try to incorporate feedback and will be accepting suggestions, collaborations and I will try to engage in conversations.

Why MixKABRN?

The general idea here is to use some of the papers and their respective strengths to propose and share a combination of them.

The papers are:
- Kolmogorov-Arnold Networks
- BitNet
- Retentive Networks
- Mixture of Experts

In an attempt at making a local, but highly functional and adaptable model that even non AI-specific devices could handle. I would like to do this to be able to reuse old hardware and allow people without access to the internet to enjoy some of the benefits of LLMs, perhaps even train them locally to translate and share information, without depending on the internet nor very complex infrastructure.

The project structure is or should be as follows, and I will try to comment anything I can to improve clarity:

my_mixkabrn_model/
│
├── my_mixkabrn_model/
│   ├── __init__.py
│   ├── mixkabrn.py  # Core implementation of the MixKABRN model (with MoE, as a base)
│   ├── bitnet_components.py  # Adapted components from BitNet
│   ├── retnet_components.py  # Adapted components from RetNet
│   └── utils.py  # Utility functions for model training and manipulation
│
├── examples/
│   └── train_text_dataset.py  # Example script for training on a text dataset (proposal of datasets)
│
├── tests/
│   └── test_mixkabrn.py  # Tests for your MoKAB model
│
├── setup.py  # Setup script for the package
└── README.md

Proposed datasets:
- M2D2
- AMPS
- MATH
- Phi-Code
- OpenQA
- Any suggestion, but trying to follow Phi or Phi-2 or similar, perhaps introduce more ideas.

### Mixture of Experts ###

- Utilizing this technique we can enhance the model's capacity, without accumulating more memory by only using a fraction of the total parameters when performing inference, yet train with all (and maybe some implementation of LoRA for finetuning on the fly?). 

### Kolmogorov - Arnold Networks ##

- By utilizing KANs instead of MLPs, and adapt them to our other structures, we can hipothesize that a smaller number of parameters would be needed for a performant model, so this is an attempt at finding a good parameter number for this types of networks. The idea is to find the least amount of parameters needed to get the most performant and small model.

- Why KANs? Instead of training on the weights, they train the activation functions, leading to a lot of potential gain in undiscovered (internal) learning, although at the cost of training cost and ease of use, although MoE and Bitnet should help with training and inference (parameter number and BitNet implementation for training).  

- Perhaps minor, but they also seem to promise to help with catastrophic forgetting, so that's that.

### BitNets ###

- BitNets take the concept of decimal accuracy and change it for only 3 options: -1, 0, and 1. By doing this, the footprint of a lot of things goes down.

- Instead of performing quantization, BitNets train from the ground up with these considerations in mind, and they showed that they can scale similarly to non-Bit Transformers, which shows that there could be promising results by using these enhancements. I believe we would use the BitLinear stuff from here.

### Retentive Networks ###

- RetNets were posted as a possible successor to transformers, specifically for LLMs, but there have also been promising results in vision models, showing that this architecture could be more than just a proposal.

- From what I gather, the argument is that retention is a mechanism similar to attention, but trying to bypass attention's complexity without losing performance (as in being actually a good replacement for attention). 

- By employing Multi Scale Retention, we can take the best of the parallelism of training, sub-quadratic inference, and promising performance (debatable how they described some of the competing architectures like RWKV) at different scales of input and training texts with their chunkwise and recurrent paradigms. (training in parallell, inference in O(1), meaning the only cost in memory is model and weights, and its respective inference cost that seems to promise remain scalably constant (hoping that this helps performance at large scales)). 

### Datasets ###
- I kinda remember these datasets as having some potential useful compatibility, while maintaining some sort of resemblance to the ideas proposed by the Phi papers, although LLAMA3 seems to think the biggest dataset is best in the end, I am still trying to kind of do this locally, but adapt it to cloud infrastructure like Huggingface, etc., so people with better harware or budget could benefit from this, though one of the backbone ideas would be local training.

### Why not (insert something)? ###

- This has just been an idea



## Installation ##

## Usage ##

## Contributing ##

## Licenses ##



## References: ##

- kindxiaoming/pykan
- kyegomez/bitnet
- deepseek-ai/deepseek-moe
- syncdoth/retnet
- kyegomez/python-packages-template

## More References: ##

@article{dai2024deepseekmoe,
  author={Damai Dai and Chengqi Deng and Chenggang Zhao and R. X. Xu and Huazuo Gao and Deli Chen and Jiashi Li and Wangding Zeng and Xingkai Yu and Y. Wu and Zhenda Xie and Y. K. Li and Panpan Huang and Fuli Luo and Chong Ruan and Zhifang Sui and Wenfeng Liang},
  title={DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models}, 
  journal   = {CoRR},
  volume    = {abs/2401.06066},
  year      = {2024},
  url       = {https://arxiv.org/abs/2401.06066},
}

@misc{sun2023retentive,
      title={Retentive Network: A Successor to Transformer for Large Language Models}, 
      author={Yutao Sun and Li Dong and Shaohan Huang and Shuming Ma and Yuqing Xia and Jilong Xue and Jianyong Wang and Furu Wei},
      year={2023},
      eprint={2307.08621},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}

@misc{2310.11453,
Author = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
Title = {BitNet: Scaling 1-bit Transformers for Large Language Models},
Year = {2023},
Eprint = {arXiv:2310.11453},
}

@misc{reid2022m2d2,
      title={M2D2: A Massively Multi-domain Language Modeling Dataset}, 
      author={Machel Reid and Victor Zhong and Suchin Gururangan and Luke Zettlemoyer},
      year={2022},
      eprint={2210.07370},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{reid2022m2d2,
      title={M2D2: A Massively Multi-domain Language Modeling Dataset}, 
      author={Machel Reid and Victor Zhong and Suchin Gururangan and Luke Zettlemoyer},
      year={2022},
      eprint={2210.07370},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
