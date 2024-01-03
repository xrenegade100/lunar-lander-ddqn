> This project was developed during the academic year 2023/2024 as part of the Artificial Intelligence course instructed by Professor Vincenzo Deufemia @ Università degli Studi di Salerno.
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<img src="https://skillicons.dev/icons?i=py,pytorch," />

## About
The main goal of the project is to develop an intelligent agent capable of autonomously learning to pilot the Lunar Lander through training with the DDQN (Double Deep Q-Learning) algorithm.<br>
PyTorch is used to implement the neural network. Other techniques used are: simulated annealing, target network soft update, buffer memory.
<br>
The environment used in this project is LunarLander-v2 from the Gymnasium library. The comprehensive documentation for the environment can be found at the following link: https://www.gymlibrary.dev/environments/box2d/lunar_lander/.<br>
<br>
In summary, the objective of the game is to land the spacecraft between the yellow flags without crashing, while optimizing fuel consumption.
This repo also includes a modified version of the environment (`src/env/lunar_lander.py`) in which penalties are increased for engine usage (to optimize fuel consumption).

## Preview
![result1704318933](https://github.com/xrenegade100/lunar-lander-ddqn/assets/11615441/723a50e8-8df3-4c77-916a-753cb9d238ef)


## Installation
1. Clone the repo and move into the downloaded directory.
```bash
$ git clone https://github.com/xrenegade100/lunar-lander-ddqn.git
$ cd lunar-lander-ddqn
```
2. Install required dependencies
```bash
pip install -r requirements.txt
```
3. **(Optional)** To utilize the modified environment (increased penalties for engine usage):
```bash
$ (lunar-lander-ddqn) > pip install -e ./src/env
```
Open `src/train.py` or `src/test.py` (based on what you want to run) and uncomment those lines:
https://github.com/xrenegade100/lunar-lander-ddqn/blob/80bbcfe19ec444e9a6c345adc32b5eaf7b03130f/src/train.py#L5-L6
https://github.com/xrenegade100/lunar-lander-ddqn/blob/80bbcfe19ec444e9a6c345adc32b5eaf7b03130f/src/train.py#L44
And comment the following:
https://github.com/xrenegade100/lunar-lander-ddqn/blob/80bbcfe19ec444e9a6c345adc32b5eaf7b03130f/src/train.py#L45

## Usage 
* To train the agent:
```bash
$ (lunar-lander-test) > cd src
$ (lunar-lander-test/src) > python train.py
```
_Note that this does not render any window to speed up the training process_
* To run a trained agent (from a file):
   1. Update the path to load the model in `src/test.py`
https://github.com/xrenegade100/lunar-lander-ddqn/blob/80bbcfe19ec444e9a6c345adc32b5eaf7b03130f/src/test.py#L10-L11
   2. Run the following command:
  ```bash
  $ (lunar-lander-test) > cd src
  $ (lunar-lander-test/src) > python test.py
  ```
## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xrenegade100"><img src="https://avatars.githubusercontent.com/u/11615441?v=4?s=100" width="100px;" alt="Antonio Scognamiglio"/><br /><sub><b>Antonio Scognamiglio</b></sub></a><br /><a href="#projectManagement-xrenegade100" title="Project Management">📆</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
