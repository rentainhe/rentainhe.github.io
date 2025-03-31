## Collection of Basic Concepts in Reinforcement Learning

<a id="top"></a>

## Contents

* [What is "Model" in RL](#what-is-model-in-rl)
* [What is the difference between Policy and Value](#what-is-the-difference-between-policy-and-value)
* [What is Model-Free and Model-Based RL](#what-is-model-free-and-model-based-rl)

---

<a id="what-is-model-in-rl"></a>
## What is "Model" in RL

* **Concept:** In RL, a **Model** refers to the agent's representation of how the environment works. It predicts what the environment will do next.
* **What it Predicts:** Specifically, a model usually predicts two things:
    1.  **State Transitions:** Given a state and an action, what is the probability of ending up in each possible next state? (P(s' | s, a))
    2.  **Rewards:** Given a state and an action (and possibly the next state), what is the expected immediate reward? (R(s, a) or R(s, a, s'))
* **Easy Way to Remember:** Think of the model as the **"rulebook" or "physics simulator"** of the RL world, as understood by the agent. It answers: *"If I do action 'a' in state 's', what will likely happen next (s') and what immediate reward will I get?"*

[Back to Top](#top)

---

<a id="what-is-the-difference-between-policy-and-value"></a>
## What is the difference between Policy and Value

These two concepts are central to how an RL agent behaves and learns.

### Policy (π)

* **Concept:** The **Policy** is the agent's strategy or behavior. It defines how the agent chooses its actions in any given state.
* **What it Does:** It maps states to actions.
    * It can be **deterministic**: In state 's', always take action 'a'.
    * It can be **stochastic**: In state 's', take action 'a' with probability P(a|s).
* **Easy Way to Remember:** The Policy is the **"What to do?"** function. It's the agent's plan or set of rules for acting.

### Value Function (V or Q)

* **Concept:** A **Value Function** predicts the expected *long-term* reward (return) an agent can get.
* **What it Does:** It estimates "how good" a situation is.
    * **State-Value Function (V(s))**: Predicts the expected total future reward starting from state 's' and then following the policy π. *How good is it to be in this state?*
    * **Action-Value Function (Q(s, a))**: Predicts the expected total future reward starting from state 's', taking action 'a', and *then* following the policy π. *How good is it to take this action in this state?*
* **Easy Way to Remember:** The Value Function is the **"How good is this?"** function. It predicts future rewards, guiding the agent towards better states/actions.

### The Key Difference

* **Policy** tells the agent **what action to take**.
* **Value** tells the agent **how good** a state or state-action pair is in terms of expected future rewards.
* Often, value functions are learned *to help find* a better policy (e.g., by always picking the action with the highest Q-value).

[Back to Top](#top)

---

<a id="what-is-model-free-and-model-based-rl"></a>
## What is Model-Free and Model-Based RL

This distinction is about *whether* the agent explicitly learns and uses a model of the environment.

### Model-Based RL

* **Concept:** The agent first learns a **model** of the environment (predicting state transitions and rewards). It then uses this learned model to **plan** ahead (e.g., by simulating potential action sequences) to figure out the best policy or value function.
* **Process:**
    1.  Interact with the environment to learn the model (the "rules").
    2.  Use the model to simulate and plan offline to find the best strategy.
* **Easy Way to Remember:** **"Learn the rules, then plan."** Like learning chess rules and predicting opponent moves to decide your move.
* **Potential Advantage:** Can be more sample efficient (learn faster from fewer real interactions) because it can "imagine" experiences using the model.

### Model-Free RL

* **Concept:** The agent learns a policy or value function **directly** from experience (trial and error) *without* explicitly building a model of how the environment works. It doesn't try to predict the next state or immediate reward.
* **Process:**
    1.  Interact with the environment.
    2.  Directly update the policy (e.g., "that action sequence worked well, do it more") or value function (e.g., "that action in that state led to a good outcome eventually") based on rewards received.
* **Easy Way to Remember:** **"Learn by doing."** Like learning to ride a bike purely through trial and error, without understanding the physics.
* **Potential Advantage:** Often simpler to implement and can work even when the environment is too complex to model accurately.

### The Key Difference

* **Model-Based** agents try to understand *how the world works* (build a model) and use that understanding to plan.
* **Model-Free** agents learn *what actions work* directly from experience, without necessarily understanding *why* they work in terms of environment dynamics.

[Back to Top](#top)