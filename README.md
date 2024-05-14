# MixKABRN
This is a draft for making the MixKABRN Neural Network, and first adapt it for training on text, and later adjust for other modalities.
I haven't picked up coding again after personal issues, so please be patient if you do not like the quality, I'll try to incorporate feedback and will be accepting suggestions, collaborations.

The general idea here is to use some of the papers and their respective strengths to propose and share a combination of them.

The papers are:
- Kolmogorov-Arnold Networks
- BitNet
- Retentive Networks
- Mixture of Experts

References:

kindxiaoming/pykan
kyegomez/bitnet
deepseek-ai/deepseek-moe
syncdoth/retnet
kyegomez/python-packages-template


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
