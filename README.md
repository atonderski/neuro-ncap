<div align="center">
    <h2>NeuroNCAP<br/>Photorealistic Closed-loop Safety Testing for Autonomous Driving
    <br/>
    <br/>
    <a href="https://research.zenseact.com/publications/neuro-ncap/"><img src="https://img.shields.io/badge/Project-Page-ffa"/></a>
    <a href="https://arxiv.org/abs/2404.07762"><img src='https://img.shields.io/badge/arXiv-Page-aff'></a>
    </h2>
</div>




https://github.com/wljungbergh/NeuroNCAP/assets/37999571/5725e2af-8215-4573-9372-c0ca8c03f5f0


This is the official repository for NeuroNCAP: Photorealistic Closed-loop Safety Testing for Autonomous Driving


## Getting started
- [How to run the evaluation](docs/how-to-run.md)
- [Framework overview](docs/framework.md)
- [Details on the evaluation protocol](docs/evaluation-protocol-overview.md)

## News <a name="news"></a>
- **`2024/10/02`** Initial code release
- **`2024/04/12`** NeuroNCAP [paper](https://arxiv.org/abs/2404.07762) published on arXiv.



## TODOs
- [ ] Gaussian Splatting version (for faster evaluation)
- [ ] VAD inference runner example
- [x] UniAD inference runner example (https://github.com/wljungbergh/UniAD)
- [x] Initial code release
- [x] Paper release


## Results

We hope to update these tables with more models in the future. If you have a model you would like to add, please open a PR.

#### NeuroNCAP score:

| Model | Avg   | Stationary | Frontal | Side  |
| ----- | ----- | ---------- | ------- | ----- |
| UniAD | 2.111 | 3.501      | 1.166   | 1.667 |

#### Collision rate (%)

| Model | Avg  | Stationary | Frontal | Side |
| ----- | ---- | ---------- | ------- | ---- |
| UniAD | 60.4 | 32.4       | 77.6    | 71.2 |

Note that the results differ slighlty from the paper due to the use of different version of NeuRAD as well as minor improvements to the simulator (better collision velocity estimation, and better controller tuning).

## Related resources
- [NeuRAD](https://github.com/georghess/neurad-studio)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [VAD](https://github.com/hustvl/VAD)

## Citation
If you find this work useful, please consider citing:
```bibtex
@article{ljungbergh2024neuroncap,
  title={NeuroNCAP: Photorealistic Closed-loop Safety Testing for Autonomous Driving},
  author={Ljungbergh, William and Tonderski, Adam and Johnander, Joakim and Caesar, Holger and {\AA}str{\"o}m, Kalle and Felsberg, Michael and Petersson, Christoffer},
  journal={European Conference on Computer Vision (ECCV)},
  year={2024}
}
