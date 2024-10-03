# NeuroNCAP Evaluation

## Scoring
The **NeuroNCAP** scoring is based on either completely avoiding collisions or minimizing the collision severity by reducing the relative velocity between the ego vehicle and the target vehicle. Formally the scoring is defined as follows:

$$
\text{score} = \begin{cases}
    5 & \text{if no collision} \\
    4 \cdot \max(0, 1 - v_i/v_r & \text{otherwise}
\end{cases}
$$
where $v_i$ is the impact speed (magnitude of the relative velocity @ impact) and $v_r$ the reference impact speed which is obtained by finding the impact speed that would have happened if no actions were taken.

## Scenarios
Your model will be evaluated across 100 random permutations of each scenario. This random permutation can involve both lateral and longitudinal shifts of the target actor. Each of the scenarios are defined by the scenario files ìn the `scenerios` folder, and they can be visualized using `neuro_ncap/visualization/scenario_visualizer.py`. The scenarios are divided into three categories: stationary, frontal, and side.

#### Stationary
For the stationary scenarios we use the following ten sequences `scene-0099`, `scene-0101`, `scene-0103`, `scene-0106`, `scene-0108`, `scene-0278`, `scene-0331`, `scene-0783`, `scene-0796`, and `scene-0966`.

#### Frontal
For the frontal scenarios we use the follwing five sequences `scene-0103`, `scene-0106`, `scene-0110`, `scene-0346`, and `scene-0923`.

#### Side
For the side scenarios we use the follwing five sequences `scene-0103`, `scene-0106`, `scene-0110`, `scene-0278`, and `scene-0921`.
