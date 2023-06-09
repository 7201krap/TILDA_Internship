### Implementation of "Safe Mutation for Deep and Recurrent Neural Networks through output gradients" by Joel Lehman et al. with some modifications.

- Official Implementation: https://github.com/uber-research/safemutations and Full paper: https://arxiv.org/abs/1712.06563

- Note that results below are the results that generated from safe_mutation_variant. However, the results is very similar to the original implementation.

- Watch the following video for implementation details (lecture delivered in Korean): https://drive.google.com/drive/folders/1GTIesw6HvqRhNw9fCnT7HrU3Yxq39TE9?usp=sharing

- See the following for the paper explanation (written in Korean): https://docs.google.com/document/d/1gFjd3-4RwQ30fUAacXsCk7DIDtWMjfhS9XjNRLo4Xec/edit?usp=sharing 

- The followings are important quotes from the paper:
  -  "A central reason is that while random mutation generally works in low dimensions, 
  a random perturbation of thousands or millions of weights will likely break existing functionality. 
  This paper proposes a solution: a family of safe mutation (SM) operators that facilitate exploration without 
  dramatically altering network behavior or requiring additional interaction with the environment. 
  The most effective SM variant scales the degree of mutation of each individual weight 
  according to the sensitivity of the network’s outputs to that weight, 
  which requires computing the gradient of outputs with respect to the weights (instead of the gradient of error, as in conventional deep learning)."
  - "A further challenge is that in deep and recurrent NNs, 
  there may be drastic differences in the sensitivity of parameters;
  for example, perturbing a weight in the layer immediately following the inputs has rippling consequences 
  as activation propagates through subsequent layers, unlike perturbation in layers nearer to the output. 
  In other words, simple mutation schemes are unlikely to tune individual parameters according to their sensitivity."
  - "The results across a variety of domains highlight the general promise of safe mutation operators, 
  in particular when guided by gradient information, which enables evolving deeper and larger recurrent networks. 
  Importantly, because SM methods generate variation without taking performance into account, 
  they can easily apply to artificial life or open-ended evolution domains, 
  where well-defined reward signals may not exist"
  - "That SM-G can now capitalize on gradient information for the purpose of safer mutation is an interesting 
  melding of concepts that previously seemed isolated in their separate paradigms. 
  SM-G is still distinct from the reward-based gradients in deep learning in that 
  it is computing a gradient entirely without reference to how it is expected to impact reward,
  but it nevertheless capitalizes on the availability of gradient computation to make informed decisions"


- See 'py_files' for implementation. For every file in 'py_files', there is a variable called 'first_run'. 
  - Set *first_run=True* if you want to see the training process and generate a new graph. The graph will be automatically saved to 'results' folder.
  - Set *first_run=False* if you want to see the visualized(=video) result.


- See 'results' for visualized graphs and weights.


- In the paper, there are 5 types of mutations.
1 through 4 are implemented and 5 and 6 are not implemented in this repository.
There is one additional type of mutation(=SM-G-FO) that I developed:
  1. Control: Entire parameter vector is perturbed with fixed-variance Gaussian noise.
  2. SM-G-FO: SafeMutation-Gradient-FirstOrder
  3. SM-G-SO: SafeMutation-Gradient-SecondOrder
  4. SM-G-SUM: SafeMutation-Gradient-Summation
  5. SM-R: SafeMutation-Rescaling
  6. SM-G-ABS: SafeMutation-Gradient-Absolute


- I compared Control, SM-G-FO, SM-G-SO, and SM-G-SUM with different environments: Cartpole, LunarLander, Maze, and BipedalWalker. 
Advantages of SM-G approach over Control could be clearly observable in Cartpole, LunarLander, and Maze environment. 
 
1. Cartpole<br>
  
|                     **Cartpole with Control**                     |            **Cartpole with SM-G-FO**             |
|:-----------------------------------------------------------------:|:------------------------------------------------:|
| <img alt="Image 1" src="safe_mutation_variant/results/cartpole ga control.png"/> | <img alt="Image 2" src="safe_mutation_variant/results/cartpole ga sm-g-fo.png"/>  |
|                     **Cartpole with SM-G-SO**                     |            **Cartpole with SM-G-SUM**            |
| <img alt="Image 3" src="safe_mutation_variant/results/cartpole ga sm-g-so.png"/> | <img alt="Image 4" src="safe_mutation_variant/results/cartpole ga sm-g-sum.png"/> |

2. LunarLander<br>

|            **LunarLander with Control**             |            **LunarLander with SM-G-FO**             |
|:---------------------------------------------------:|:---------------------------------------------------:|
| <img alt="Image 5" src="safe_mutation_variant/results/lundarlander ga control.png"/> | <img alt="Image 6" src="safe_mutation_variant/results/lunarlander ga sm-g-fo.png"/>  |
|            **LunarLander with SM-G-SO**             |            **LunarLander with SM-G-SUM**            |
| <img alt="Image 7" src="safe_mutation_variant/results/lunarlander ga sm-g-so.png"/>  | <img alt="Image 8" src="safe_mutation_variant/results/lunarlander ga sm-g-sum.png"/> |

3. Maze<br>

|             **Maze with Control**              |             **Maze with SM-G-FO**             |
|:----------------------------------------------:|:---------------------------------------------:|
|  <img alt="Image 9" src="safe_mutation_variant/results/maze ga control.png"/>   | <img alt="Image 10" src="safe_mutation_variant/results/Maze ga sm-g-fo.png"/>  |
|             **Maze with SM-G-SO**              |            **Maze with SM-G-SUM**             |
| <img alt="Image 11" src="safe_mutation_variant/results/Maze ga sm-g-so.png"/>   | <img alt="Image 12" src="safe_mutation_variant/results/maze ga sm-g-sum.png"/> |

4. BipedalWalker (Doesn't work very well to both control and SM-G-X approach)<br>

|                              **BipedalWalker with Control**                              |          **BipedalWalker with SM-G-FO**          |
|:----------------------------------------------------------------------------------------:|:------------------------------------------------:|
| Graph Not Available (Checked that its performance is not greater than SM-G-SUM approach) | <img alt="Image 13" src="safe_mutation_variant/results/bipedal ga sm-g-fo.png"/>  |
|                              **BipedalWalker with SM-G-SO**                              |         **BipedalWalker with SM-G-SUM**          |
|                     <img alt="Image 14" src="safe_mutation_variant/results/Bipedal ga sm-g-so.png"/>                      | <img alt="Image 15" src="safe_mutation_variant/results/Bipedal ga sm-g-sum.png"/> |

