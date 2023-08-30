# Hierarchical Option Critic
**Work in Progress**

This is a reimplementation of Hierarchical Option Critic, introduced in Learning Abstract Options [(Riemer et al., 2018)](https://proceedings.neurips.cc/paper/2018/hash/cdf28f8b7d14ab02d12a2329d71e4079-Abstract.html).

## Why Hierarchical Option Critic?
Hierarchical Option Critic is an end-to-end gradient-based skill hierarchy where the number of levels and number of skills at each level of the hierarchy must be manually specified. The Hierarchical Option Critic is an extension of the line of work that produced the [Option Critic](https://ojs.aaai.org/index.php/AAAI/article/view/10916/10775). Where an agent using Option Critic defines one layer of skills over primitive actions, Hierarchical Option Critic allows the agent to produce an skill hierarchy of any number of levels.

Multi-level skill hierarchies are the future of reinforcement learning. Consider the behaviours used to make an English breakfast. We might follow the following sequence of behaviours at the highest level of abstraction: _make fried eggs_, _make toast_, _cook sausages_, _heat baked beans_. The behaviour (or skill) _make fried eggs_ is further decomposed into _grab ingredients and pan_, _heat pan with oil_, _crack egg into pan_, _wait for egg to cook_, _put fried egg on plate_, which together define a set of skills at a lower level of abstraction. The skill _crack egg into pan_ consists of another sequence of lower-level behaviours which eventually decompose into the individual contractions in the various muscles we might use to _crack egg into pan_. Altogether, this simplified decomposition of the overall task requires at least three levels of skills in the skill hierarchy (not including primitive actions) where the top-most skill is _make an English breakfast_. 

Learning modular skills within a hierarchy allows us humans to reuse previously learned behaviours in new situations instead of relearning the same skill over and over again for each new situation we need to use the same skill. In addition, skills abstract away from the finer details of task completion if we have already learned the constituent behaviours of said task. In other words, we don't have to relearn how to _crack eggs_ every morning and we don't have to calculate the speed and direction our finger should move to activate the toaster when _making toast_. It is clear that this type of decomposition is common to several tasks we might be interested in automating. 

## Possible Extensions

Since the authors published Learning Abstract Options, there have been a few newer features added to the Option Critic research direction.

1. **Interest functions**.
2. **Off-Policy Option Updates**
Combining these with the Hierarchical Option Critic should allow the agent to learn even faster on new tasks.

A less straightforward challenge is to relax the conditions of specifying the number of levels and number of skills per level required by the agent. An agent that can infer how deep the skill hierarchy should be and how many skills to define at each level of abstraction would be much more adaptive. 
