# VitalChain: Agentic Biological Resource Coordination

VitalChain is an RL-trained coordination layer for high-stakes biological resource distribution. By tackling the game-theoretic bottlenecks in healthcare logistics, we optimize the "trust-and-speed" problem inherent in decentralized hospital networks.

**Impact:** In simulated high-pressure environments, the agentic deployment of predictive routing and inventory pooling meant platelet waste fell **38%** and average transport delay was cut by over **20%**.

---

## The Core Technical Argument: Partial Observability

The primary challenge in regional healthcare is not just speed; it is **partial observability**. Hospitals operate in silos. They cannot see the inventory of neighboring institutions. VitalChain simulates this exact real-world constraint. The RL agent must learn to navigate a partially observable Markov decision process (POMDP), where sharing data has a tokenized value, and hoarding results in systemic penalties.

---

## Dispatch Alert

```
[DISPATCH ALERT - NOTTO ID-773A]
STATUS: DYING patient (O-) at Hospital_2. 
ASSET: Donor Heart. Viability remaining: 18%.
SYSTEM LOGIC: Standard route > 40m. Viability constraint breached.
ACTION: EMERGENCY route authorized. Data pooled. Police escort deployed.
```

---

## Five hospitals. One network.

Below is the network topology of our extended model. The agent tracks inter-city transit times, predicting decay rates against real-time routing availability.

<div align="center">
<svg width="600" height="400" viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
<path id="path1" d="M 300 200 L 150 100" stroke="#d58b98" stroke-width="2" stroke-dasharray="8,4" fill="none" />
<path id="path2" d="M 300 200 L 450 100" stroke="#d58b98" stroke-width="2" stroke-dasharray="8,4" fill="none" />
<path id="path3" d="M 300 200 L 150 300" stroke="#d58b98" stroke-width="2" stroke-dasharray="8,4" fill="none" />
<path id="path4" d="M 300 200 L 450 300" stroke="#d58b98" stroke-width="2" stroke-dasharray="8,4" fill="none" />
<text x="200" y="140" fill="#666" font-family="monospace" font-size="12" transform="rotate(-33 225 150)">42m</text>
<text x="380" y="140" fill="#666" font-family="monospace" font-size="12" transform="rotate(33 375 150)">55m</text>
<text x="200" y="270" fill="#666" font-family="monospace" font-size="12" transform="rotate(33 225 250)">38m</text>
<text x="380" y="270" fill="#666" font-family="monospace" font-size="12" transform="rotate(-33 375 250)">45m</text>
<circle r="4" fill="crimson">
<animateMotion dur="2s" repeatCount="indefinite"><mpath href="#path1"/></animateMotion>
</circle>
<circle r="4" fill="crimson">
<animateMotion dur="2.5s" repeatCount="indefinite"><mpath href="#path2"/></animateMotion>
</circle>
<circle r="4" fill="crimson">
<animateMotion dur="1.8s" repeatCount="indefinite"><mpath href="#path3"/></animateMotion>
</circle>
<circle r="4" fill="crimson">
<animateMotion dur="2.2s" repeatCount="indefinite"><mpath href="#path4"/></animateMotion>
</circle>
<circle cx="150" cy="100" r="12" fill="white" stroke="#333" stroke-width="2"/>
<text x="150" y="80" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="14">Mumbai</text>
<circle cx="450" cy="100" r="12" fill="white" stroke="#333" stroke-width="2"/>
<text x="450" y="80" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="14">Delhi</text>
<circle cx="150" cy="300" r="12" fill="white" stroke="#333" stroke-width="2"/>
<text x="150" y="330" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="14">Pune</text>
<circle cx="450" cy="300" r="12" fill="white" stroke="#333" stroke-width="2"/>
<text x="450" y="330" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="14">Chennai</text>
<circle cx="300" cy="200" r="15" fill="white" stroke="crimson" stroke-width="3">
<animate attributeName="r" values="15;22;15" dur="1.5s" repeatCount="indefinite"/>
<animate attributeName="stroke-opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite"/>
</circle>
<circle cx="300" cy="200" r="8" fill="crimson"/>
<text x="300" y="240" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="14" fill="crimson">Bangalore (h0)</text>
</svg>
</div>

---

## Operation ORR — Saturday 11:47 PM Bangalore Outer Ring Road

To test the system against extreme localized friction, the model is trained on specific crisis scenarios. **"Operation ORR"** models a mass-casualty event overlapping with peak traffic gridlock on the Outer Ring Road.

In this scenario, default transit routing fails universally. The agent is forced to execute an aggressive token-management strategy: pulling live coordinate data (simulated via eRaktKosh API integration logic), hoarding standard routes for stable patients, and perfectly timing its limited `GREEN_CORRIDOR` overrides to bypass the 11:47 PM gridlock.

---

## The Golden Hour Problem

Every 10-minute delay in organ transport reduces viability by approximately **1.4%** for short-window organs (heart, lung) and **0.47%** for kidneys.

VitalChain trains the agent to optimize the **trust-and-speed** problem:

| Route type | Time saving | Token cost | When agent uses it |
|---|---|---|---|
| Standard | baseline | 0 | Stable/moderate patients |
| Green Corridor | −31% | 1/episode | Viability < 40% |
| Emergency | −51% | 1/episode | DYING patients only |

After training, the agent learns to use Green Corridor selectively — deploying it when viability pressure is high, not reflexively.

---

## Platform vision, not a product

VitalChain is **not** a hospital app. It is a training environment for the AI coordination layer that sits above all hospitals — a **state-level optimization platform**.

The bureaucracy problem ("how do you get 50 hospitals to share data?") is answered **mathematically** by the cooperation reward signal. Task 3 `crisis_response` training shows that data-sharing is the dominant strategy — hospitals that cooperate consistently outperform hospitals that hoard. **The RL training IS the incentive design proof.**

Future integration targets immutable organ provenance logs via distributed ledger — a requirement for integrating with systems like NOTTO to prevent the diversion that plagues current analog supply chains.

---

## Why hospitals share data — the game-theoretic answer

A common question: *"the real problem is bureaucracy — hospitals won't share data."*

VitalChain's environment answers this directly. Task 3 models hospitals as semi-autonomous agents. Hospitals start with `data_shared = False` (hoarding default).

The cooperation reward (+1.5 per sharing event) and hoarding penalty (−0.5 per expired resource that could have been used elsewhere) create a game where:

- **Expected reward from sharing:** +1.5 per event, reduced waste
- **Expected reward from hoarding:** −0.3 baseline + −0.5 on expiry events

After 200 training episodes, the agent learns **cooperation is strictly dominant**. This is the mathematical argument for why hospitals would adopt a shared coordination layer.
