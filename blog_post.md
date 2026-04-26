# The Last 45 Minutes: Teaching AI to Save Lives When Logistics Fail

I still remember the headline. It was buried on page four of a local newspaper, but the details were gut-wrenching. 

A donor heart had become available. A patient in critical condition was waiting just 40 kilometers away. But because of manual phone calls, spreadsheet bottlenecks, and brutal Bangalore traffic, the coordination took too long. By the time the ambulance arrived, the heart had exceeded its 6-hour cold ischemia time limit. It was no longer viable. 

**The patient died not because of a lack of organs, but because of a lack of infrastructure.**

Every day in India, 18 people die waiting for an organ. Hundreds more suffer from localized shortages of specific blood types, even while hospitals just a few districts over throw away expired platelets. When you look at these tragedies closely, you realize a hard truth: this isn’t a medical problem. **It is a logistics and routing problem.**

And that is exactly the kind of problem AI is built to solve.

---

## The Birth of VitalChain

I realized that what we needed wasn't another hospital management dashboard. We needed an intelligent, state-level protocol—a "TCP/IP for biological logistics."

I started building **VitalChain** for the OpenEnv 2026 Hackathon. The goal: create a Reinforcement Learning (RL) environment that simulates a high-stakes, multi-hospital network, and train an AI agent to act as the ultimate resource coordinator.

But writing the simulation was only half the battle. How do you teach a Large Language Model (LLM) the brutal triage mathematics of life and death? How do you teach it that saving a stable patient is a *failure* if a dying patient was ignored?

## Training the Mind of a Coordinator

We decided to use **GRPO** (Group Relative Policy Optimization) to train a lightweight model (`SmolLM2-135M`) via LoRA. But instead of giving the AI hard-coded "if-then" rules, we used **reward shaping**.

We built seven composable reward rubrics. Every action the agent takes is judged against these vectors:

1. 🚨 **Patient Outcome:** Massive rewards for treating DYING patients. Negative rewards for treating STABLE patients if others are more urgent.
2. ⏳ **Waste Penalty:** Heavy penalties if an organ expires in storage.
3. 🧬 **Compatibility:** A strict, unbreakable negative penalty for violating ABO or HLA blood-type matching.
4. 🛑 **Inaction Penalty:** The harshest lesson. The agent receives a devastating penalty (-0.333) if it chooses to `wait` while a dying patient has compatible resources available.
5. 🤝 **Anti-Hoarding:** Penalties for a hospital hoarding rare blood types while a patient at a neighboring hospital needs it.

### The "Aha!" Moment

When training started, the agent acted like a paralyzed bureaucrat. It chose to `wait` almost constantly, racking up massive inaction penalties. 

But then, around **Training Step 31**, something incredible happened.

The agent realized that waiting was catastrophic. It began issuing `allocate` commands. The loss function dropped into the negatives (a massive signal that it found a winning strategy), and the gradient updates spiked. 

By Step 100, the agent completely stopped waiting when lives were on the line. By Step 200, it was actively routing organs via Green Corridors and prioritizing the exact patients the medical rubrics dictated. **It learned that in a biological network, cooperation is the dominant strategy.**

## Looking Forward

We are currently running a 400-step training job to push the model's capabilities even further (this blog post will be updated with final metrics soon!). 

But the core proof-of-concept is already here. VitalChain proves that we can train open-weight LLMs to handle multi-constraint, life-or-death resource allocation under extreme time pressure. 

If this AI could be integrated into real-world systems like India's *eRaktKosh* blood network or the *NOTTO* organ registry, it wouldn't just make logistics more efficient. 

It would buy people the last 45 minutes they need to live.

---
*Built for the OpenEnv Hackathon India 2026 — Theme #1: Multi-Agent Interactions*
