### Implementation of "Efficiently Evolving Programs through the Search for Novelty" by Joel Lehman and Kenneth O. Stanley with some modifications. 
- In the paper, the authors noted that "... It is important to note that this conception 
of novelty is the same as it was in neuroevolution 
(i.e. rewarding novel behaviors, not novel genotypes) ...".
I'm actually not sure about the true meaning of 'novel behaviors'. 
What is the difference between 'novel behaviors' and 'novel genotypes'?
- I think they wanted to say that novel behaviors could be satisfied by 
introducing some conditions. For example, the agent should perform 
behavior1(=action1) for condition1, behavior2(=action2) for condition2, and so on 
- Considering the above, it seems like my implementation does not fully 
incorporate the actual meaning of 'behaviors', since the agent(=red square)'s 
genotype is represented by a vector of possible actions where the actions 
are 'L = turn left', 'R = turn right', 'M = move forward'. 
For example a single agent's genotype is represented by the following:
['R', 'M', 'M', 'L', 'R', 'M', 'L', 'M', 'M'].
- Nevertheless, my implementation contains the main idea which is using the novelty for fitness and choosing offsprings based on the novelty.
- There are four files that you can run:
  1. easy_maze_novelty.py (to run: python easy_maze_novelty.py)
     1. Final agent positions<br>
        ![easy maze novelty final agent positions](results/easy%20maze%20novelty%20final%20agent%20pos.png)
     2. Best distance history<br>
        ![easy maze novelty history](results/easy%20maze%20novelty%20history.png)
  2. easy_maze_original.py (to run: python easy_maze_original.py)
     1. Final agent positions <br>
        ![easy maze original final agent positions](results/easy%20maze%20original%20final%20agent%20pos.png)
     2. Best distance history<br>
        ![easy maze original final agent positions](results/easy%20maze%20original%20final%20history.png)
  3. hard_maze_novelty.py (to run: python hard_maze_novelty.py)
     1. Final agent positions<br>
        ![hard maze novelty final agent positions](results/hard%20maze%20novelty%20final%20agent%20pos.png)
     2. Best distance history<br>
        ![hard maze novelty history](results/hard%20maze%20novelty%20history.png)
  4. hard_maze_original.py (to run: python hard_maze_original.py)
     1. Final agent positions<br>
        ![hard maze original final agent positions](results/hard%20maze%20original%20final%20agent%20pos.png)
     2. Best distance history<br>
        ![hard maze original history](results/hard%20maze%20original%20history.png)

- It is easy to catch the difference between 'novelty' and 'original' by running the 3rd and 4th file.
- Last update (2023-06-03)
