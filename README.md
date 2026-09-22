<div align="center">

# pH-Strips for Selective Forgetting: A Blunt but Fast Diagnostic Baseline for Machine Unlearning

[![Venue:CVPR 2026](https://img.shields.io/badge/Venue-CVPR%202026%20-blue)](https://openaccess.thecvf.com/content/CVPR2026/papers/Qian_pH-Strips_for_Selective_Forgetting_A_Blunt_but_Fast_Diagnostic_Baseline_CVPR_2026_paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


## Abstract
Machine Unlearning (MU), erasing undesirable content from Artificial Intelligence (AI) models, plays an essential role in developing safe and trustworthy AI systems. Despite notable advances, the baseline MU methods rely on retraining from scratch without the data to be removed, which is computationally expensive and financially prohibitive. To address this challenge, we propose a simple yet efficient training-free and retain-set-free MU algorithm designed explicitly as a diagnostic baseline: Machine Unlearning pH-Test (MUpHT). It is designed to serve as a practical evaluation reference for future MU methods. Our method eliminates the low-dimensional subspaces associated with undesirable concepts from the space spanned by the model's weight vectors, thereby rendering the model ''blind" to these undesirable contents. Additionally, we further offer a retain-aware variant to handle entangled features by leveraging a generalized Rayleigh quotient over the undesirable and retain sets, enabling an efficient tradeoff between preserving retained knowledge and suppressing undesirable knowledge. Our method enables evaluation of MU across diverse visual tasks, including concept erasure for classification, image generation, and multimodal applications. By producing an unlearned model instantly from only a few samples, our method serves as a quick litmus test for MU.

# MUpHT for Classification
This is the official repository for MUpHT for Clasification. The code structure of this project is adapted from the [Sparse Unlearn](https://github.com/OPTML-Group/Unlearn-Sparse) codebase.


## Scripts
  *  MUpHT
  ```bash
  python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn MUpHT --class_to_replace 1 ${forgetting class}
    ```


## BibTeX
```
@inproceedings{
  phstrips,
  title={pH-Strips for Selective Forgetting: A Blunt but Fast Diagnostic Baseline for Machine Unlearning},
  author={Chengyao Qian and Jing Wu and Trung Le and Dinh Phung and Mehrtash Harandi},
  booktitle={Conference on Computer Vision and Pattern Recognition 2026},
  year={2026},
}
```

## Acknowledgements
This repository makes liberal use of code from [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency), [Selective Amnesia](https://github.com/clear-nus/selective-amnesia) and [ESD](https://github.com/rohitgandikota/erasing/tree/main).
