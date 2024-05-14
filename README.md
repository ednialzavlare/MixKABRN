# MixKABRN

## Table of Contents

1. [Introduction](#introduction)
2. [Table of Contents](#table-of-contents)
3. [Description](#description)
4. [Why MixKABRN?](#why-mixkabrn)
5. [Project Structure](#project-structure)
6. [Proposed Datasets](#proposed-datasets)
7. [Core Components](#core-components)
   - [Mixture of Experts](#mixture-of-experts)
   - [Kolmogorov-Arnold Networks (KANs)](#kolmogorov-arnold-networks-kans)
   - [BitNets](#bitnets)
   - [Retentive Networks](#retentive-networks)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)
12. [References](#references)
13. [More References](#more-references)


## To Do ##
- Make a basic implementation that works
- Mostly everything and fix what doesn't work (still very fresh).
- Remove placeholders and pseudocode (right now it mostly is pseudocode)
- Tests
- Open pretrained models and weights will be available, but a training pipeline needs to be set up.
(I'm trying to set up a local server for training and inference, later on this).

## Description ##

This is a repo for the MixKABRN Neural Network (Mixture of Kolmogorov-Arnold Bit Retentive Networks), and first adapt it for training on text, and later adjust for other modalities.

Please be patient if you do not like the quality, I'll try to incorporate feedback and will be accepting suggestions, collaborations and I will try to engage in conversation.

Why MixKABRN?

The general idea here is to use some of the papers and their respective strengths to propose and share a combination of them.

The main papers are:
- Kolmogorov-Arnold Networks
- BitNet
- Retentive Networks
- Mixture of Experts

And some other are:
- Textbooks are all you need
- MATH
- M2D2

In an attempt at making a local, but highly functional and adaptable model that even non AI-specific devices could handle. I would like to do this to be able to reuse old hardware and allow people without access to the internet to enjoy some of the benefits of LLMs, perhaps even train them locally to translate and share information, without depending on the internet nor very complex infrastructure.

The project structure is roughly as follows, and I will try to comment anything I can to improve clarity, or try suggestions:

![image](https://github.com/ednial0zavlare/MixKABRN/assets/125082787/ff17a47b-6b38-46ad-91b9-f9405df6c106)


### Proposed datasets: ###
- M2D2
- AMPS
- MATH
- Phi-Code
- OpenQA
- Any suggestion, but trying to follow Phi or Phi-2 or similar, perhaps introduce more ideas.

- This might not be worth it but, translating the datasets or extrapolating translated datasets could help with reinforcing ideas already seen, but encoded in different ways to take advantage of the relations from the KAN parts.(maybe the same could be said for new info, but by relating it to previous ones, it might be easier to learn?)

### Mixture of Experts ###

- Utilizing this technique we can enhance the model's capacity, without accumulating more memory by only using a fraction of the total parameters when performing inference, yet train with all (and maybe some implementation of LoRA for finetuning on the fly?). 

### Kolmogorov - Arnold Networks ##

- By utilizing KANs instead of MLPs, and adapt them to our other structures, we can hipothesize that a smaller number of parameters would be needed for a performant model, so this is an attempt at finding a good parameter number for this types of networks. The idea is to find the least amount of parameters needed to get the most performant and small model.

- Why KANs? Instead of training on the weights, they train the activation functions, leading to a lot of potential gain in undiscovered (internal) learning, although at the cost of training cost and ease of use, although MoE and Bitnet should help with training and inference (parameter number and BitNet implementation for training).  

- Perhaps minor, but they also seem to promise to help with catastrophic forgetting, so that's that.

### BitNets ###

- BitNets take the concept of decimal accuracy and changed it for only 3 options: -1, 0, and 1. By doing this, the footprint of a lot of things goes down.

- Instead of performing quantization, BitNets train from the ground up with these considerations in mind, and they showed that they can scale similarly to non-Bit Transformers, which shows that there could be promising results by using these enhancements. I believe we would use the BitLinear stuff from here.

### Retentive Networks ###

- RetNets were posted as a possible successor to transformers, specifically for LLMs, but there have also been promising results in vision models, showing that this architecture could be more than just a proposal.

- From what I gather, the argument is that retention is a mechanism similar to attention, but trying to bypass attention's complexity without losing performance (as in being actually a good replacement for attention). 

- By employing Multi Scale Retention, we can take the best of the parallelism of training, sub-quadratic inference, and promising performance (debatable how they described some of the competing architectures like RWKV) at different scales of input and training texts with their chunkwise and recurrent paradigms. (training in parallell, inference in O(1), meaning the only cost in memory is model and weights, and its respective inference cost that seems to promise remain scalably constant (hoping that this helps performance at large scales)). 

### Datasets ###
- I kinda remember these datasets as having some potential useful compatibility, while maintaining some sort of resemblance to the ideas proposed by the Phi papers, although LLAMA3 seems to think the biggest dataset is best in the end, I am still trying to kind of do this locally, but adapt it to cloud infrastructure like Huggingface, etc., so people with better harware or budget could benefit from this, though one of the backbone ideas would be local training.

### Why not (insert something)? ###

- This has just been an idea, and if you have some that would help that's cool, but maybe we just settle with something and build up on it? Further discussion is open, maybe this is a Wild Goose chase and with your idea it can change course to something completable?

## Installation ##

### NON working example: ###

```
git clone https://github.com/ednial0zavlare/MixKABRN.git
cd MixKABRN
pip install -r requirements.txt
```


## Usage ##

### NON working example: ###

```python train.py --dataset M2D2 --model mixkabrn```


## Contributing ##

I believe this could help, although suggestions are open :)

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Licenses

This project is licensed under the MIT License - see the LICENSE.md file for details.

## References: ##

- [kindxiaoming/pykan](https://github.com/kindxiaoming/pykan) - Repository for PyKan project.
- [kyegomez/bitnet](https://github.com/kyegomez/bitnet) - Repository for BitNet project.
- [deepseek-ai/deepseek-moe](https://github.com/deepseek-ai/deepseek-moe) - Repository for DeepSeek MoE project.
- [syncdoth/retnet](https://github.com/syncdoth/retnet) - Repository for RetNet project.
- [kyegomez/python-packages-template](https://github.com/kyegomez/python-packages-template) - Template for creating Python packages.


## More References:

```bibtex

@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}

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

@misc{fan2023rmt,
      title={RMT: Retentive Networks Meet Vision Transformers}, 
      author={Qihang Fan and Huaibo Huang and Mingrui Chen and Hongmin Liu and Ran He},
      year={2023},
      eprint={2309.11523},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{abdin2024phi3,
      title={Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone}, 
      author={Marah Abdin and Sam Ade Jacobs and Ammar Ahmad Awan and Jyoti Aneja and Ahmed Awadallah and Hany Awadalla and Nguyen Bach and Amit Bahree and Arash Bakhtiari and Harkirat Behl and Alon Benhaim and Misha Bilenko and Johan Bjorck and Sébastien Bubeck and Martin Cai and Caio César Teodoro Mendes and Weizhu Chen and Vishrav Chaudhary and Parul Chopra and Allie Del Giorno and Gustavo de Rosa and Matthew Dixon and Ronen Eldan and Dan Iter and Amit Garg and Abhishek Goswami and Suriya Gunasekar and Emman Haider and Junheng Hao and Russell J. Hewett and Jamie Huynh and Mojan Javaheripi and Xin Jin and Piero Kauffmann and Nikos Karampatziakis and Dongwoo Kim and Mahoud Khademi and Lev Kurilenko and James R. Lee and Yin Tat Lee and Yuanzhi Li and Chen Liang and Weishung Liu and Eric Lin and Zeqi Lin and Piyush Madan and Arindam Mitra and Hardik Modi and Anh Nguyen and Brandon Norick and Barun Patra and Daniel Perez-Becker and Thomas Portet and Reid Pryzant and Heyang Qin and Marko Radmilac and Corby Rosset and Sambudha Roy and Olatunji Ruwase and Olli Saarikivi and Amin Saied and Adil Salim and Michael Santacroce and Shital Shah and Ning Shang and Hiteshi Sharma and Xia Song and Masahiro Tanaka and Xin Wang and Rachel Ward and Guanhua Wang and Philipp Witte and Michael Wyatt and Can Xu and Jiahang Xu and Sonali Yadav and Fan Yang and Ziyi Yang and Donghan Yu and Chengruidong Zhang and Cyril Zhang and Jianwen Zhang and Li Lyna Zhang and Yi Zhang and Yue Zhang and Yunan Zhang and Xiren Zhou},
      year={2024},
      eprint={2404.14219},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{hu2021lora,
      title={LoRA: Low-Rank Adaptation of Large Language Models}, 
      author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
      year={2021},
      eprint={2106.09685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
