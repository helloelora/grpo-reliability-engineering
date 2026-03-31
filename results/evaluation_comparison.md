# GRPO V3 Evaluation Results: Base vs Fine-tuned

## Overall Summary

| Split | Base | Fine-tuned | Delta | Interpretation |
|-------|------|------------|-------|----------------|
| A. Training set (57 questions) | 58.3% (133/228) | 57.9% (132/228) | -0.4% | In-distribution performance |
| B. Hard holdout (29 questions) | 29.3% (34/116) | 32.8% (38/116) | +3.4% | Generalization to unseen hard problems |
| C. Master holdout (169 questions) | 71.9% (486/676) | 71.2% (481/676) | -0.7% | Catastrophic forgetting check |

---

## A. Training set (57 questions)

**Base:** All correct=16, Partial=32, All wrong=9
**Fine-tuned:** All correct=13, Partial=35, All wrong=9

### Per-question comparison

| # | Target | Base | FT | Change | Question (truncated) |
|---|--------|------|-----|--------|----------------------|
| 1 | 0.316 | 4/4 | 3/4 | DEGRADED | A baseball player has a current batting average of .250 (pro... |
| 2 | 0.99992525 | 4/4 | 4/4 | same | For a certain type of airplane to fly, at least two out of i... |
| 3 | 0.2057 | 4/4 | 4/4 | same | Suppose that a population of components follows the life-dis... |
| 4 | 8333.0 | 2/4 | 3/4 | IMPROVED | A reliability engineer is comparing two suppliers for a crit... |
| 5 | 0.00109 | 3/4 | 4/4 | IMPROVED | A power supply unit has a constant failure rate of λ = 0.004... |
| 6 | 0.607 | 0/4 | 0/4 | same | A data center has backup generators that activate automatica... |
| 7 | 0.61 | 4/4 | 4/4 | same | A data center cooling system consists of 4 independent chill... |
| 8 | 0.9978 | 4/4 | 3/4 | DEGRADED | A manufacturing facility operates a critical production line... |
| 9 | 0.769 | 4/4 | 4/4 | same | A power supply unit has a lifetime that follows a lognormal ... |
| 10 | 493.7 | 2/4 | 0/4 | DEGRADED | A reliability engineer is analyzing failure data for hydraul... |
| 11 | 0.9818 | 4/4 | 4/4 | same | A data center uses a redundant power supply system with thre... |
| 12 | 0.915 | 2/4 | 2/4 | same | A satellite communication system has a critical transponder ... |
| 13 | 0.9712 | 1/4 | 2/4 | IMPROVED | A satellite communication system has a critical transponder ... |
| 14 | 0.549 | 4/4 | 4/4 | same | A manufacturing quality control system uses a 3-out-of-5 red... |
| 15 | 0.2915 | 2/4 | 3/4 | IMPROVED | Industrial pumps used in chemical processing plants have lif... |
| 16 | 0.9955 | 4/4 | 3/4 | DEGRADED | A telecommunications company is designing a redundant power ... |
| 17 | 0.945 | 0/4 | 0/4 | same | A telecommunications company is designing a redundant power ... |
| 18 | 0.42 | 3/4 | 3/4 | same | A quality control engineer is using a single sampling plan f... |
| 19 | 13624.0 | 4/4 | 3/4 | DEGRADED | A data center operates three independent cooling systems (A,... |
| 20 | 4436.5 | 4/4 | 4/4 | same | A power supply unit has a lifetime that follows a Weibull di... |
| 21 | 0.696 | 4/4 | 3/4 | DEGRADED | A reliability engineer is analyzing field failure data for a... |
| 22 | 0.581327 | 2/4 | 3/4 | IMPROVED | A safety-critical system consists of two subsystems in serie... |
| 23 | 0.6466 | 0/4 | 1/4 | IMPROVED | A hydraulic system consists of three subsystems in series. S... |
| 24 | 574.326 | 0/4 | 0/4 | same | A fleet manager monitors a fleet of identical vehicles whose... |
| 25 | 0.6479 | 2/4 | 1/4 | DEGRADED | A safety system consists of two independent subsystems in se... |
| 26 | 5336.6 | 3/4 | 4/4 | IMPROVED | An engineer is conducting an accelerated life test on cerami... |
| 27 | 0.70956 | 0/4 | 0/4 | same | An engineer is designing a standby redundancy system with on... |
| 28 | 0.86676 | 0/4 | 1/4 | IMPROVED | A chemical processing plant has a safety system consisting o... |
| 29 | 0.82715 | 2/4 | 3/4 | IMPROVED | A chemical plant has a safety system consisting of two subsy... |
| 30 | 0.66582 | 3/4 | 1/4 | DEGRADED | An engineer is performing a Bayesian reliability analysis on... |
| 31 | 0.9344 | 2/4 | 3/4 | IMPROVED | A safety system consists of two identical subsystems in para... |
| 32 | 0.117647 | 2/4 | 3/4 | IMPROVED | A critical safety system consists of two identical subsystem... |
| 33 | 0.96982 | 2/4 | 2/4 | same | A critical safety system consists of two identical subsystem... |
| 34 | 0.77243 | 0/4 | 1/4 | IMPROVED | A fleet manager observes failure data for a critical engine ... |
| 35 | 0.535946 | 1/4 | 1/4 | same | A chemical processing system has two identical pumps in acti... |
| 36 | 0.4917 | 4/4 | 3/4 | DEGRADED | A fleet of 10 identical turbines is placed on test. Their li... |
| 37 | 0.6958 | 2/4 | 3/4 | IMPROVED | A chemical processing plant has a critical safety system wit... |
| 38 | 0.90317 | 3/4 | 3/4 | same | A chemical plant has a safety system consisting of two subsy... |
| 39 | 0.3479 | 2/4 | 0/4 | DEGRADED | A system consists of two subsystems in series. Subsystem A i... |
| 40 | 0.77907 | 2/4 | 4/4 | IMPROVED | A chemical processing plant has a safety system consisting o... |
| 41 | 0.9731 | 4/4 | 2/4 | DEGRADED | A chemical plant has a safety system consisting of two subsy... |
| 42 | 0.6058 | 3/4 | 4/4 | IMPROVED | A critical safety system consists of two subsystems in serie... |
| 43 | 1598.6 | 3/4 | 3/4 | same | A fleet of 10 identical devices is placed on a reliability t... |
| 44 | 0.957843 | 1/4 | 1/4 | same | A fleet of 10 identical turbine engines is placed on a relia... |
| 45 | 0.31322 | 4/4 | 2/4 | DEGRADED | A critical safety system consists of two identical subsystem... |
| 46 | 1084.56 | 0/4 | 1/4 | IMPROVED | A fleet of 10 identical turbines is placed on a reliability ... |
| 47 | 0.4871 | 2/4 | 3/4 | IMPROVED | A critical safety system consists of two identical component... |
| 48 | 16.86 | 0/4 | 0/4 | same | A fleet of 20 identical turbine blades is placed on a life t... |
| 49 | 0.686475 | 2/4 | 2/4 | same | A system consists of two subsystems in series. Subsystem A i... |
| 50 | 0.98037 | 3/4 | 4/4 | IMPROVED | A chemical processing plant has a safety system consisting o... |
| 51 | 57.154 | 3/4 | 4/4 | IMPROVED | A fleet of 10 identical turbine blades is placed on a reliab... |
| 52 | 0.930068 | 2/4 | 0/4 | DEGRADED | A safety-critical system consists of two subsystems in serie... |
| 53 | 0.6254 | 1/4 | 0/4 | DEGRADED | A fleet of 10 identical turbine blades is placed on a life t... |
| 54 | 0.9874 | 3/4 | 2/4 | DEGRADED | A fleet manager operates 50 identical trucks whose engines h... |
| 55 | 0.98728 | 2/4 | 2/4 | same | A fleet of 20 identical turbine blades is put on a life test... |
| 56 | 0.49128 | 1/4 | 2/4 | IMPROVED | A fleet manager observes failure data for a critical engine ... |
| 57 | 0.98331 | 4/4 | 3/4 | DEGRADED | A repairable power generation unit has two independent failu... |

**Improved:** 20 questions | **Degraded:** 16 questions | **Same:** 21 questions

#### Improved questions (fine-tuned > base)

- **Q4** (2/4 -> 3/4): target=8333.0
  - Q: A reliability engineer is comparing two suppliers for a critical electronic component. A sample of 2...
  - Gen 3: base=1000.0 (X) -> ft=8333.33 (V)

- **Q5** (3/4 -> 4/4): target=0.00109
  - Q: A power supply unit has a constant failure rate of λ = 0.004 failures per 1000 hours. The unit opera...
  - Gen 0: base=0.0 (X) -> ft=0.00109 (V)

- **Q13** (1/4 -> 2/4): target=0.9712
  - Q: A satellite communication system has a critical transponder subsystem consisting of 5 independent ch...
  - Gen 1: base=0.9116 (X) -> ft=0.9999 (V)
  - Gen 2: base=0.0198 (X) -> ft=0.9857 (V)

- **Q15** (2/4 -> 3/4): target=0.2915
  - Q: Industrial pumps used in chemical processing plants have lifetimes that follow a Weibull distributio...
  - Gen 0: base=0.244 (X) -> ft=0.292 (V)

- **Q22** (2/4 -> 3/4): target=0.581327
  - Q: A safety-critical system consists of two subsystems in series. Subsystem A is a 2-out-of-3 redundant...
  - Gen 3: base=0.336169 (X) -> ft=0.582647 (V)

- **Q23** (0/4 -> 1/4): target=0.6466
  - Q: A hydraulic system consists of three subsystems in series. Subsystem A has two identical components ...
  - Gen 1: base=0.6998 (X) -> ft=0.6537 (V)

- **Q26** (3/4 -> 4/4): target=5336.6
  - Q: An engineer is conducting an accelerated life test on ceramic capacitors using temperature as the ac...
  - Gen 0: base=9.66 (X) -> ft=5335.0 (V)

- **Q28** (0/4 -> 1/4): target=0.86676
  - Q: A chemical processing plant has a safety system consisting of three identical pressure relief valves...
  - Gen 0: base=0.3971 (X) -> ft=0.886 (V)

- **Q29** (2/4 -> 3/4): target=0.82715
  - Q: A chemical plant has a safety system consisting of two subsystems in series. Subsystem A is a 2-out-...
  - Gen 0: base=0.1323 (X) -> ft=0.829 (V)

- **Q31** (2/4 -> 3/4): target=0.9344
  - Q: A safety system consists of two identical subsystems in parallel (both must fail for system failure)...
  - Gen 0: base=0.6775 (X) -> ft=0.9796 (V)
  - Gen 1: base=0.398 (X) -> ft=0.9738 (V)

- **Q32** (2/4 -> 3/4): target=0.117647
  - Q: A critical safety system consists of two identical subsystems in parallel (both must fail for system...
  - Gen 3: base=17.0 (X) -> ft=0.1176 (V)

- **Q34** (0/4 -> 1/4): target=0.77243
  - Q: A fleet manager observes failure data for a critical engine component believed to follow a Weibull d...
  - Gen 3: base=0.4 (X) -> ft=0.8107 (V)

- **Q37** (2/4 -> 3/4): target=0.6958
  - Q: A chemical processing plant has a critical safety system with the following architecture: two identi...
  - Gen 1: base=0.9429 (X) -> ft=0.7018 (V)
  - Gen 3: base=0.5616 (X) -> ft=0.714 (V)

- **Q40** (2/4 -> 4/4): target=0.77907
  - Q: A chemical processing plant has a safety system consisting of two identical pressure relief valves a...
  - Gen 2: base=4.0 (X) -> ft=0.7778 (V)
  - Gen 3: base=-0.25 (X) -> ft=0.7788 (V)

- **Q42** (3/4 -> 4/4): target=0.6058
  - Q: A critical safety system consists of two subsystems in series. Subsystem 1 is a 2-out-of-3 redundant...
  - Gen 2: base=0.1336 (X) -> ft=0.6057 (V)

- **Q46** (0/4 -> 1/4): target=1084.56
  - Q: A fleet of 10 identical turbines is placed on a reliability test. The lifetimes are assumed to follo...
  - Gen 2: base=1533.82 (X) -> ft=1084.56 (V)

- **Q47** (2/4 -> 3/4): target=0.4871
  - Q: A critical safety system consists of two identical components in parallel (both must fail for system...
  - Gen 0: base=0.41 (X) -> ft=0.487 (V)
  - Gen 2: base=0.4405 (X) -> ft=0.487 (V)

- **Q50** (3/4 -> 4/4): target=0.98037
  - Q: A chemical processing plant has a safety system consisting of three independent subsystems. Subsyste...
  - Gen 2: base=0.8366 (X) -> ft=0.9818 (V)

- **Q51** (3/4 -> 4/4): target=57.154
  - Q: A fleet of 10 identical turbine blades is placed on a reliability life test. The failure times (in t...
  - Gen 1: base=50.03 (X) -> ft=56.7 (V)

- **Q56** (1/4 -> 2/4): target=0.49128
  - Q: A fleet manager observes failure data for a critical engine component. From prior engineering analys...
  - Gen 1: base=0.445 (X) -> ft=0.5087 (V)
  - Gen 2: base=1.4799999999999998e-57 (X) -> ft=0.489 (V)

#### Degraded questions (fine-tuned < base)

- **Q1** (4/4 -> 3/4): target=0.316
  - Q: A baseball player has a current batting average of .250 (probability of a hit p = 0.250). In a game,...
  - Gen 2: base=0.3164 (V) -> ft=256.0 (X)

- **Q8** (4/4 -> 3/4): target=0.9978
  - Q: A manufacturing facility operates a critical production line that requires at least 2 out of 4 paral...
  - Gen 1: base=0.9976 (V) -> ft=0.767 (X)

- **Q10** (2/4 -> 0/4): target=493.7
  - Q: A reliability engineer is analyzing failure data for hydraulic pumps. A sample of 150 pumps is put o...
  - Gen 0: base=481.0 (V) -> ft=396.47 (X)
  - Gen 3: base=493.0 (V) -> ft=376.0 (X)

- **Q16** (4/4 -> 3/4): target=0.9955
  - Q: A telecommunications company is designing a redundant power supply system for a cell tower. The syst...
  - Gen 2: base=0.9955 (V) -> ft=0.9332 (X)

- **Q19** (4/4 -> 3/4): target=13624.0
  - Q: A data center operates three independent cooling systems (A, B, and C) in parallel to maintain serve...
  - Gen 3: base=13614.0 (V) -> ft=7645.0 (X)

- **Q21** (4/4 -> 3/4): target=0.696
  - Q: A reliability engineer is analyzing field failure data for a critical automotive component. Historic...
  - Gen 2: base=0.695 (V) -> ft=0.628 (X)

- **Q25** (2/4 -> 1/4): target=0.6479
  - Q: A safety system consists of two independent subsystems in series. Subsystem A is a 2-out-of-3 redund...
  - Gen 0: base=0.636 (V) -> ft=0.8186 (X)
  - Gen 3: base=0.6223 (V) -> ft=0.4572 (X)

- **Q30** (3/4 -> 1/4): target=0.66582
  - Q: An engineer is performing a Bayesian reliability analysis on a new turbine blade design. The prior d...
  - Gen 0: base=0.68 (V) -> ft=0.507 (X)
  - Gen 3: base=0.6657 (V) -> ft=0.761 (X)

- **Q36** (4/4 -> 3/4): target=0.4917
  - Q: A fleet of 10 identical turbines is placed on test. Their lifetimes follow a Weibull distribution wi...
  - Gen 3: base=0.49 (V) -> ft=0.31 (X)

- **Q39** (2/4 -> 0/4): target=0.3479
  - Q: A system consists of two subsystems in series. Subsystem A is a 2-out-of-3 redundant configuration w...
  - Gen 1: base=0.3523 (V) -> ft=0.399 (X)
  - Gen 2: base=0.345 (V) -> ft=0.542 (X)

- **Q41** (4/4 -> 2/4): target=0.9731
  - Q: A chemical plant has a safety system consisting of two subsystems in series. Subsystem 1 is a 2-out-...
  - Gen 1: base=0.9491 (V) -> ft=0.8345 (X)
  - Gen 2: base=0.9826 (V) -> ft=0.00892 (X)

- **Q45** (4/4 -> 2/4): target=0.31322
  - Q: A critical safety system consists of two identical subsystems in parallel (both must fail for system...
  - Gen 1: base=0.3175 (V) -> ft=0.0295 (X)
  - Gen 2: base=0.3125 (V) -> ft=0.9973 (X)

- **Q52** (2/4 -> 0/4): target=0.930068
  - Q: A safety-critical system consists of two subsystems in series. Subsystem A is a 2-out-of-3 redundant...
  - Gen 1: base=0.9145 (V) -> ft=0.8209 (X)
  - Gen 2: base=0.9405 (V) -> ft=0.8138 (X)

- **Q53** (1/4 -> 0/4): target=0.6254
  - Q: A fleet of 10 identical turbine blades is placed on a life test. The failure times (in thousands of ...
  - Gen 2: base=0.603 (V) -> ft=0.904 (X)

- **Q54** (3/4 -> 2/4): target=0.9874
  - Q: A fleet manager operates 50 identical trucks whose engines have Weibull-distributed lifetimes with s...
  - Gen 0: base=0.9876 (V) -> ft=0.0 (X)
  - Gen 1: base=0.9871 (V) -> ft=0.619 (X)

- **Q57** (4/4 -> 3/4): target=0.98331
  - Q: A repairable power generation unit has two independent failure modes that act as competing risks. Fa...
  - Gen 1: base=0.97153 (V) -> ft=0.92713 (X)

---

## B. Hard holdout (29 questions)

**Base:** All correct=6, Partial=6, All wrong=17
**Fine-tuned:** All correct=5, Partial=10, All wrong=14

### Per-question comparison

| # | Target | Base | FT | Change | Question (truncated) |
|---|--------|------|-----|--------|----------------------|
| 1 | 6.848e-05 | 0/4 | 0/4 | same | An engineer is running an accelerated life test (ALT) on cap... |
| 2 | 0.60681 | 0/4 | 0/4 | same | A fleet of 10 identical turbine engines is placed on a relia... |
| 3 | 0.51201 | 0/4 | 0/4 | same | A chemical plant has a safety system consisting of two subsy... |
| 4 | 0.35145 | 4/4 | 1/4 | DEGRADED | A critical safety system consists of two identical subsystem... |
| 5 | 0.995 | 2/4 | 0/4 | DEGRADED | A fleet of 20 identical turbine blades is placed on an accel... |
| 6 | 0.06516 | 0/4 | 3/4 | IMPROVED | A fleet manager observes failure data for a critical engine ... |
| 7 | 983.7 | 0/4 | 0/4 | same | A fleet of 20 identical turbine blades is placed on a reliab... |
| 8 | 8336.0 | 0/4 | 0/4 | same | A fleet of 10 identical turbines is placed on a reliability ... |
| 9 | 0.95272 | 1/4 | 2/4 | IMPROVED | A critical safety system consists of two subsystems in serie... |
| 10 | 300202.0 | 0/4 | 0/4 | same | An accelerated life test is conducted on ceramic capacitors ... |
| 11 | 2.0046e-06 | 0/4 | 0/4 | same | An accelerated life test is conducted on a component whose l... |
| 12 | 26.47 | 0/4 | 1/4 | IMPROVED | A fleet of 20 identical turbine blades is placed on a reliab... |
| 13 | 214670000.0 | 0/4 | 0/4 | same | A fleet of 50 identical turbine blades is placed on an accel... |
| 14 | 0.00198 | 4/4 | 4/4 | same | A fleet of 15 identical turbine blades is placed on an accel... |
| 15 | 0.3379 | 1/4 | 0/4 | DEGRADED | A fleet manager is monitoring turbine blade lifetimes. From ... |
| 16 | 0.9717 | 0/4 | 2/4 | IMPROVED | A fleet of 20 identical turbine blades is placed on an accel... |
| 17 | 0.985113 | 4/4 | 4/4 | same | A repairable system has two independent failure modes operat... |
| 18 | 14273.0 | 0/4 | 0/4 | same | A fleet of 20 identical turbine blades is placed on a reliab... |
| 19 | 0.89502 | 3/4 | 2/4 | DEGRADED | A fleet of 10 identical devices is placed on a life test. Th... |
| 20 | 0.5169 | 0/4 | 1/4 | IMPROVED | A chemical processing facility has a safety shutdown system ... |
| 21 | 6699.0 | 0/4 | 0/4 | same | A fleet of 20 identical turbine blades is placed on a life t... |
| 22 | 0.7299 | 2/4 | 2/4 | same | A chemical plant has a safety system consisting of two subsy... |
| 23 | 0.8678 | 0/4 | 1/4 | IMPROVED | A chemical plant has a safety system composed of two identic... |
| 24 | 0.5527 | 1/4 | 0/4 | DEGRADED | A fleet manager monitors two independent failure modes (Mode... |
| 25 | 0.7711 | 4/4 | 4/4 | same | A critical safety system consists of two subsystems in serie... |
| 26 | 5604.0 | 0/4 | 0/4 | same | An accelerated life test is conducted on a mechanical actuat... |
| 27 | 8.0129 | 0/4 | 3/4 | IMPROVED | A fleet of 10 identical turbines is placed on test. Their li... |
| 28 | 0.99131 | 4/4 | 4/4 | same | A chemical processing plant has a pressure relief system wit... |
| 29 | 0.75 | 4/4 | 4/4 | same | A chemical plant has a safety shutdown system consisting of ... |

**Improved:** 7 questions | **Degraded:** 5 questions | **Same:** 17 questions

#### Improved questions (fine-tuned > base)

- **Q6** (0/4 -> 3/4): target=0.06516
  - Q: A fleet manager observes failure data for a critical engine component from 20 units placed on test. ...
  - Gen 0: base=0.0211 (X) -> ft=0.064 (V)
  - Gen 2: base=0.0229 (X) -> ft=0.066 (V)
  - Gen 3: base=0.0248 (X) -> ft=0.0664 (V)

- **Q9** (1/4 -> 2/4): target=0.95272
  - Q: A critical safety system consists of two subsystems in series. Subsystem A is a 2-out-of-3 redundant...
  - Gen 2: base=0.458 (X) -> ft=0.9713 (V)

- **Q12** (0/4 -> 1/4): target=26.47
  - Q: A fleet of 20 identical turbine blades is placed on a reliability life test. The test is Type-II rig...
  - Gen 2: base=13.04 (X) -> ft=27.46 (V)

- **Q16** (0/4 -> 2/4): target=0.9717
  - Q: A fleet of 20 identical turbine blades is placed on an accelerated life test at a high stress level ...
  - Gen 1: base=0.582 (X) -> ft=0.99896 (V)
  - Gen 3: base=0.005 (X) -> ft=0.977 (V)

- **Q20** (0/4 -> 1/4): target=0.5169
  - Q: A chemical processing facility has a safety shutdown system consisting of three independent subsyste...
  - Gen 1: base=0.6771 (X) -> ft=0.508 (V)

- **Q23** (0/4 -> 1/4): target=0.8678
  - Q: A chemical plant has a safety system composed of two identical pressure relief valves arranged in pa...
  - Gen 0: base=1.0 (X) -> ft=0.8953 (V)

- **Q27** (0/4 -> 3/4): target=8.0129
  - Q: A fleet of 10 identical turbines is placed on test. Their lifetimes (in thousands of hours) follow a...
  - Gen 0: base=6.1134 (X) -> ft=7.7632 (V)
  - Gen 2: base=12.2307 (X) -> ft=8.0025 (V)
  - Gen 3: base=7.4076 (X) -> ft=8.0025 (V)

#### Degraded questions (fine-tuned < base)

- **Q4** (4/4 -> 1/4): target=0.35145
  - Q: A critical safety system consists of two identical subsystems in parallel (both must fail for system...
  - Gen 0: base=0.3611 (V) -> ft=0.8324 (X)
  - Gen 1: base=0.3525 (V) -> ft=0.3759 (X)
  - Gen 2: base=0.358 (V) -> ft=0.6569 (X)

- **Q5** (2/4 -> 0/4): target=0.995
  - Q: A fleet of 20 identical turbine blades is placed on an accelerated life test at a high stress level....
  - Gen 0: base=0.9979 (V) -> ft=0.895 (X)
  - Gen 2: base=1.0 (V) -> ft=0.0 (X)

- **Q15** (1/4 -> 0/4): target=0.3379
  - Q: A fleet manager is monitoring turbine blade lifetimes. From historical data on a previous blade desi...
  - Gen 1: base=0.33 (V) -> ft=1.0 (X)

- **Q19** (3/4 -> 2/4): target=0.89502
  - Q: A fleet of 10 identical devices is placed on a life test. The devices have exponentially distributed...
  - Gen 3: base=0.885 (V) -> ft=0.9935 (X)

- **Q24** (1/4 -> 0/4): target=0.5527
  - Q: A fleet manager monitors two independent failure modes (Mode 1 and Mode 2) acting as competing risks...
  - Gen 2: base=0.5488 (V) -> ft=0.494 (X)

---

## C. Master holdout (169 questions)

**Base:** All correct=105, Partial=29, All wrong=35
**Fine-tuned:** All correct=101, Partial=37, All wrong=31

### Per-question comparison

| # | Target | Base | FT | Change | Question (truncated) |
|---|--------|------|-----|--------|----------------------|
| 1 | 1.0 | 4/4 | 4/4 | same | Let $F(t) = 1 - (1+t)^{-1}$ for $0 \le t \le \infty$. Find t... |
| 2 | 0.2841 | 4/4 | 3/4 | DEGRADED | Three assembly plants produce the same parts. Plant A produc... |
| 3 | 2.4 | 4/4 | 4/4 | same | An automatic misting system in an agricultural study follows... |
| 4 | 15.0 | 4/4 | 4/4 | same | An automatic misting system in an agricultural study follows... |
| 5 | 0.6703 | 2/4 | 1/4 | DEGRADED | An automatic misting system in an agricultural study follows... |
| 6 | 1132.0 | 4/4 | 3/4 | DEGRADED | Assume that the lifetime follows a lognormal distribution, f... |
| 7 | 5323.0 | 0/4 | 0/4 | same | Suppose we want to be 90% confident of meeting an MTTF of at... |
| 8 | 3106.0 | 0/4 | 1/4 | IMPROVED | Suppose we want to be 60% confident of meeting an MTTF of at... |
| 9 | 2023.0 | 0/4 | 0/4 | same | Suppose we want to be 60% confident of meeting an MTTF of at... |
| 10 | 0.0769 | 4/4 | 4/4 | same | 50 devices are placed on stress for 168 hours. The probabili... |
| 11 | 2.5 | 4/4 | 4/4 | same | 50 devices are placed on stress for 168 hours. The probabili... |
| 12 | 0.9231 | 4/4 | 4/4 | same | 50 devices are placed on stress for 168 hours. The probabili... |
| 13 | 103.0 | 3/4 | 3/4 | same | Using the Goal Seek function, find the necessary sample size... |
| 14 | 175.0 | 0/4 | 0/4 | same | Using the Goal Seek function, find the necessary sample size... |
| 15 | 32.0 | 4/4 | 4/4 | same | An electronic component acts to transform an input signal of... |
| 16 | 6.0 | 4/4 | 4/4 | same | An electronic component acts to transform an input signal of... |
| 17 | 0.09121 | 4/4 | 3/4 | DEGRADED | An electronic component acts to transform an input signal of... |
| 18 | 0.873 | 4/4 | 4/4 | same | An electronic card contains three components: A, B, and C. C... |
| 19 | 24.0 | 4/4 | 4/4 | same | Consider the case of four interarrival times. How many permu... |
| 20 | 0.042 | 0/4 | 0/4 | same | Consider the case of four interarrival times (labeled 1, 2, ... |
| 21 | 2.0 | 2/4 | 1/4 | DEGRADED | Consider the exponential distribution with MTTF = 50,000 hou... |
| 22 | 5268.0 | 4/4 | 4/4 | same | Consider the exponential distribution with MTTF = 50,000 hou... |
| 23 | 34657.0 | 4/4 | 4/4 | same | Consider the exponential distribution with MTTF = 50,000 hou... |
| 24 | 1.0 | 4/4 | 4/4 | same | A baseball player has a current batting average of .250 (pro... |
| 25 | 0.866 | 4/4 | 3/4 | DEGRADED | A baseball player has a current batting average of .250 (pro... |
| 26 | 0.422 | 4/4 | 4/4 | same | A baseball player has a current batting average of .250 (pro... |
| 27 | 0.0039 | 3/4 | 3/4 | same | A baseball player has a current batting average of .250 (pro... |
| 28 | 0.684 | 4/4 | 4/4 | same | A baseball player has a current batting average of .250 (pro... |
| 29 | 2773.0 | 0/4 | 0/4 | same | Suppose 500 units are tested for 1000 hours with no fails. E... |
| 30 | 1386.0 | 0/4 | 0/4 | same | Suppose 500 units are tested for 1000 hours with no fails. C... |
| 31 | 54167.0 | 0/4 | 1/4 | IMPROVED | A space satellite is exposed to severe radiation. The engine... |
| 32 | 0.99465 | 4/4 | 4/4 | same | A system consists of two boxes. Each box consists of three i... |
| 33 | 23.0 | 2/4 | 3/4 | IMPROVED | 1000 randomly selected lines of code are inspected. 23 bugs ... |
| 34 | 23717.0 | 0/4 | 1/4 | IMPROVED | Determine the nearly minimum sample size (allowing at most 1... |
| 35 | 22554.0 | 0/4 | 0/4 | same | We have 300 units to test, and we want to be 80% confident t... |
| 36 | 6752.0 | 0/4 | 0/4 | same | We have 300 units to test, and we want to be 60% confident t... |
| 37 | 0.3935 | 4/4 | 4/4 | same | Suppose that a population of components follows the life-dis... |
| 38 | 0.7062 | 4/4 | 4/4 | same | Suppose that a population of components follows the life-dis... |
| 39 | 0.3127 | 4/4 | 4/4 | same | Suppose that a population of components follows the life-dis... |
| 40 | 78.7 | 4/4 | 4/4 | same | Suppose that a population of components follows the life-dis... |
| 41 | 62.54 | 4/4 | 4/4 | same | Suppose that a population of components follows the life-dis... |
| 42 | 0.2424 | 4/4 | 3/4 | DEGRADED | A memory chip lifetime for soft errors follows the exponenti... |
| 43 | 0.2424 | 2/4 | 2/4 | same | A memory chip lifetime for soft errors follows the exponenti... |
| 44 | 0.03285 | 2/4 | 0/4 | DEGRADED | In an experiment to compare different treatments, the old me... |
| 45 | 0.09253 | 0/4 | 0/4 | same | In an experiment to compare different treatments, the old me... |
| 46 | 0.0016 | 0/4 | 0/4 | same | The qualification requirements allow a maximum of two failur... |
| 47 | 0.0106 | 0/4 | 0/4 | same | The qualification requirements allow a maximum of two failur... |
| 48 | 132.0 | 0/4 | 0/4 | same | The AQL field average failure rate for 24,000 hours is 50 FI... |
| 49 | 180.0 | 0/4 | 1/4 | IMPROVED | The lot acceptance criteria allows a maximum of three failur... |
| 50 | 0.029 | 0/4 | 0/4 | same | The lot acceptance criteria allows a maximum of three failur... |
| 51 | 4742.1 | 0/4 | 0/4 | same | A manufacturer is conducting accelerated life testing on a n... |
| 52 | 17.0 | 1/4 | 0/4 | DEGRADED | A manufacturer wants to demonstrate an MTTF of at least 5,00... |
| 53 | 0.9724 | 4/4 | 4/4 | same | A parallel redundant system consists of 3 independent compon... |
| 54 | 3.0 | 1/4 | 2/4 | IMPROVED | A radar system consists of a critical power amplifier compon... |
| 55 | 0.36 | 4/4 | 4/4 | same | A software development team is tracking defects in two diffe... |
| 56 | 0.924 | 1/4 | 3/4 | IMPROVED | A data center has 5 independent cooling units arranged in a ... |
| 57 | 0.07 | 4/4 | 4/4 | same | A quality control inspector is examining a batch of electron... |
| 58 | 0.998 | 4/4 | 4/4 | same | A parallel redundant system consists of 3 independent compon... |
| 59 | 0.9823 | 4/4 | 4/4 | same | A critical aerospace component follows a Weibull distributio... |
| 60 | 0.368 | 2/4 | 0/4 | DEGRADED | A system has a linearly increasing hazard rate given by h(t)... |
| 61 | 3169000.0 | 0/4 | 2/4 | IMPROVED | A semiconductor device has a lifetime that follows a lognorm... |
| 62 | 56.0 | 4/4 | 4/4 | same | A reliability test is conducted on 8 identical components un... |
| 63 | 35.0 | 4/4 | 4/4 | same | A reliability test is conducted on 8 identical components la... |
| 64 | 3750.0 | 4/4 | 4/4 | same | A reliability engineer is comparing two suppliers for a crit... |
| 65 | 0.875 | 4/4 | 4/4 | same | A reliability engineer is testing components from Supplier B... |
| 66 | 0.01 | 2/4 | 1/4 | DEGRADED | A manufacturer monitors the defect rate in production batche... |
| 67 | 0.4605 | 0/4 | 0/4 | same | A quality control inspector has a box containing 20 electron... |
| 68 | 0.0067 | 3/4 | 4/4 | IMPROVED | A power supply unit has a constant failure rate of λ = 0.004... |
| 69 | 92346.0 | 0/4 | 0/4 | same | A manufacturer conducts an accelerated life test on 300 LED ... |
| 70 | 0.5 | 3/4 | 4/4 | IMPROVED | A communication system component has a reliability function ... |
| 71 | 2.0 | 4/4 | 4/4 | same | A communication system component has a reliability function ... |
| 72 | 0.211 | 4/4 | 4/4 | same | A communication system component has a reliability function ... |
| 73 | 0.0621 | 0/4 | 0/4 | same | Electronic control units in automotive applications are foun... |
| 74 | 23790.0 | 2/4 | 0/4 | DEGRADED | Electronic control units in automotive applications are foun... |
| 75 | 0.9625 | 4/4 | 4/4 | same | A quality control system uses two independent inspection met... |
| 76 | 0.251 | 4/4 | 4/4 | same | A data center has backup generators that activate automatica... |
| 77 | 3.0 | 4/4 | 4/4 | same | A data center has backup generators that activate automatica... |
| 78 | 33700.0 | 4/4 | 4/4 | same | A semiconductor device has a lognormal failure distribution ... |
| 79 | 39511.0 | 0/4 | 0/4 | same | A reliability test is conducted on 50 identical electronic m... |
| 80 | 0.989 | 4/4 | 4/4 | same | A manufacturing system consists of 5 independent subsystems ... |
| 81 | 9.0 | 4/4 | 4/4 | same | A sensor system processes incoming signals where the output ... |
| 82 | 7.7 | 4/4 | 3/4 | DEGRADED | A component follows a lognormal distribution for its time-to... |
| 83 | 1.16 | 4/4 | 4/4 | same | A component follows a lognormal distribution for its time-to... |
| 84 | 500.0 | 4/4 | 4/4 | same | A component follows a lognormal distribution for its time-to... |
| 85 | 45.0 | 0/4 | 0/4 | same | A reliability test plan requires testing n units to failure ... |
| 86 | 0.9818 | 4/4 | 3/4 | DEGRADED | A satellite system consists of 3 redundant power modules con... |
| 87 | 0.757 | 4/4 | 4/4 | same | A mechanical component has a lifetime that follows a Weibull... |
| 88 | 0.407 | 3/4 | 4/4 | IMPROVED | A mechanical component has a lifetime that follows a Weibull... |
| 89 | 0.099 | 4/4 | 4/4 | same | A telecommunications company monitors the reliability of fib... |
| 90 | 0.374 | 4/4 | 4/4 | same | A data center cooling system consists of 4 independent chill... |
| 91 | 2.83 | 4/4 | 4/4 | same | A power supply unit has a lifetime that follows a lognormal ... |
| 92 | 243.6 | 4/4 | 4/4 | same | Consider a reliability test where five identical components ... |
| 93 | 52.47 | 4/4 | 4/4 | same | Consider a reliability test where five identical components ... |
| 94 | 0.004105 | 2/4 | 4/4 | IMPROVED | Consider a reliability test where five identical components ... |
| 95 | 173.1 | 2/4 | 2/4 | same | A manufacturer conducts accelerated life testing on electron... |
| 96 | 0.956 | 4/4 | 4/4 | same | A telecommunications server has a constant failure rate of 0... |
| 97 | 0.0104 | 4/4 | 4/4 | same | A telecommunications server has a constant failure rate of 0... |
| 98 | 666.67 | 3/4 | 4/4 | IMPROVED | A telecommunications server has a constant failure rate of 0... |
| 99 | 0.9912 | 4/4 | 4/4 | same | A telecommunications server has a constant failure rate of 0... |
| 100 | 0.9101 | 4/4 | 4/4 | same | A critical avionics system uses a hybrid redundancy configur... |
| 101 | 9398.3 | 0/4 | 0/4 | same | An aerospace company is conducting life testing on a critica... |
| 102 | 11425.0 | 0/4 | 0/4 | same | An aerospace company is conducting life testing on a critica... |
| 103 | 0.0001272 | 1/4 | 1/4 | same | An aerospace company is conducting life testing on a new hyd... |
| 104 | 5040.0 | 3/4 | 4/4 | IMPROVED | A manufacturing facility produces circuit boards and conduct... |
| 105 | 210.0 | 4/4 | 4/4 | same | A manufacturing facility produces circuit boards and conduct... |
| 106 | 0.9667 | 0/4 | 0/4 | same | A manufacturing facility produces circuit boards and conduct... |
| 107 | 120.0 | 4/4 | 4/4 | same | A quality control engineer is studying the arrival pattern o... |
| 108 | 2500.0 | 4/4 | 4/4 | same | A telecommunications company is testing the reliability of f... |
| 109 | 45.0 | 4/4 | 4/4 | same | A manufacturing quality control engineer is studying the rel... |
| 110 | 0.209 | 4/4 | 4/4 | same | A manufacturing facility experiences machine breakdowns that... |
| 111 | 6.0 | 4/4 | 4/4 | same | A manufacturing facility experiences machine breakdowns that... |
| 112 | 2.449 | 0/4 | 2/4 | IMPROVED | A manufacturing facility experiences machine breakdowns that... |
| 113 | 0.337 | 4/4 | 4/4 | same | A manufacturing facility experiences machine breakdowns that... |
| 114 | 0.9981 | 4/4 | 4/4 | same | A data center uses a redundant power supply system with inde... |
| 115 | 0.6065 | 4/4 | 4/4 | same | A critical navigation system uses a 2-out-of-4 redundancy co... |
| 116 | 0.566 | 3/4 | 4/4 | IMPROVED | A telecommunications network uses a redundant server configu... |
| 117 | 52429.0 | 0/4 | 0/4 | same | A semiconductor manufacturer is evaluating the reliability o... |
| 118 | 145947.0 | 0/4 | 1/4 | IMPROVED | A semiconductor manufacturer is evaluating the reliability o... |
| 119 | 13590.0 | 4/4 | 4/4 | same | A telecommunications satellite has a critical transponder co... |
| 120 | 53174.0 | 4/4 | 4/4 | same | A telecommunications satellite has a critical transponder co... |
| 121 | 0.256 | 4/4 | 4/4 | same | A telecommunications satellite has a critical transponder co... |
| 122 | 0.9656 | 4/4 | 4/4 | same | A data center uses a redundant cooling system with 4 identic... |
| 123 | 0.3915 | 4/4 | 4/4 | same | A data center operates a critical storage system with 5 inde... |
| 124 | 0.9711 | 4/4 | 4/4 | same | A satellite communication system has a critical transponder ... |
| 125 | 11104.0 | 2/4 | 1/4 | DEGRADED | A manufacturer produces industrial control units whose lifet... |
| 126 | 2.08 | 2/4 | 1/4 | DEGRADED | A manufacturer produces industrial control units whose lifet... |
| 127 | 4000.0 | 3/4 | 3/4 | same | A component manufacturer claims their new microprocessor has... |
| 128 | 0.4495 | 4/4 | 4/4 | same | A production batch contains 30 semiconductor chips destined ... |
| 129 | 0.613 | 4/4 | 4/4 | same | A telecommunications base station has 5 independent backup g... |
| 130 | 0.982 | 4/4 | 3/4 | DEGRADED | A wind turbine blade monitoring system uses a 2-out-of-4 vot... |
| 131 | 14.0 | 4/4 | 4/4 | same | Customer service calls arrive at a technical support center ... |
| 132 | 0.67 | 4/4 | 4/4 | same | A satellite communications system has a dual-redundant power... |
| 133 | 1250.0 | 4/4 | 4/4 | same | A data center uses a 3-out-of-5 redundancy configuration for... |
| 134 | 0.999712 | 4/4 | 4/4 | same | A telecommunications company uses a redundant server system ... |
| 135 | 0.9208 | 4/4 | 4/4 | same | Industrial pumps used in chemical processing plants have lif... |
| 136 | 2916.0 | 0/4 | 1/4 | IMPROVED | Industrial pumps used in chemical processing plants have lif... |
| 137 | 0.8272 | 4/4 | 3/4 | DEGRADED | Industrial pumps used in chemical processing plants have lif... |
| 138 | 0.0747 | 4/4 | 4/4 | same | A manufacturing facility produces automotive sensors using t... |
| 139 | 44311.0 | 4/4 | 4/4 | same | A telecommunications satellite contains a critical oscillato... |
| 140 | 0.3023 | 4/4 | 4/4 | same | A telecommunications satellite contains a critical oscillato... |
| 141 | 3.0 | 3/4 | 2/4 | DEGRADED | A telecommunications satellite contains a critical oscillato... |
| 142 | 0.4493 | 4/4 | 4/4 | same | A telecommunications satellite contains a critical oscillato... |
| 143 | 3603600.0 | 4/4 | 4/4 | same | A semiconductor manufacturing facility operates a fleet of 1... |
| 144 | 0.209 | 4/4 | 4/4 | same | A data center monitors server failures across its infrastruc... |
| 145 | 0.4493 | 4/4 | 4/4 | same | A data center monitors server failures across its infrastruc... |
| 146 | 0.4493 | 3/4 | 0/4 | DEGRADED | A data center monitors server failures across its infrastruc... |
| 147 | 0.3779 | 4/4 | 4/4 | same | A manufacturing facility produces circuit boards and tests t... |
| 148 | 182.0 | 4/4 | 4/4 | same | A manufacturer conducts accelerated life testing on electron... |
| 149 | 6.0 | 3/4 | 2/4 | DEGRADED | A telecommunications company is designing a redundant power ... |
| 150 | 0.2866 | 4/4 | 4/4 | same | A telecommunications company is designing a redundant power ... |
| 151 | 0.9025 | 4/4 | 4/4 | same | A system consists of three independent subsystems connected ... |
| 152 | 49505.0 | 4/4 | 4/4 | same | A system consists of three independent subsystems connected ... |
| 153 | 0.487 | 4/4 | 3/4 | DEGRADED | A power distribution system for a manufacturing facility use... |
| 154 | 0.2173 | 4/4 | 4/4 | same | An aerospace company designs a critical avionics control sys... |
| 155 | 78.0 | 0/4 | 0/4 | same | A quality control engineer is designing a single sampling pl... |
| 156 | 0.13 | 2/4 | 2/4 | same | A quality control engineer is designing a single sampling pl... |
| 157 | 45.0 | 4/4 | 4/4 | same | A temperature sensor produces an output voltage V that is re... |
| 158 | 0.4724 | 4/4 | 4/4 | same | A data center operates three independent cooling systems (A,... |
| 159 | 4635.0 | 0/4 | 0/4 | same | A data center operates three independent cooling systems (A,... |
| 160 | 0.9038 | 4/4 | 4/4 | same | A power supply unit has a lifetime that follows a Weibull di... |
| 161 | 0.5929 | 4/4 | 4/4 | same | A power supply unit has a lifetime that follows a Weibull di... |
| 162 | 12.0 | 4/4 | 4/4 | same | A chemical processing plant monitors the concentration of a ... |
| 163 | 0.9818 | 4/4 | 4/4 | same | A satellite communication system consists of three redundant... |
| 164 | 0.9393 | 4/4 | 4/4 | same | A satellite communication system consists of three redundant... |
| 165 | 0.988 | 4/4 | 4/4 | same | A satellite communication system consists of three redundant... |
| 166 | 24.0 | 0/4 | 0/4 | same | A telecommunications company wants to demonstrate that a fib... |
| 167 | 17.0 | 4/4 | 3/4 | DEGRADED | A software testing team is conducting defect analysis on a n... |
| 168 | 0.0001315 | 0/4 | 1/4 | IMPROVED | A semiconductor manufacturer is performing reliability quali... |
| 169 | 8.78 | 4/4 | 4/4 | same | A reliability engineer is testing a new type of LED light bu... |

**Improved:** 19 questions | **Degraded:** 23 questions | **Same:** 127 questions

#### Improved questions (fine-tuned > base)

- **Q8** (0/4 -> 1/4): target=3106.0
  - Q: Suppose we want to be 60% confident of meeting an MTTF of at least 2,000,000 hours (in other words, ...
  - Gen 2: base=917.0 (X) -> ft=3000.0 (V)

- **Q31** (0/4 -> 1/4): target=54167.0
  - Q: A space satellite is exposed to severe radiation. The engineers designed the system to operate if at...
  - Gen 2: base=25000.0 (X) -> ft=54166.67 (V)

- **Q33** (2/4 -> 3/4): target=23.0
  - Q: 1000 randomly selected lines of code are inspected. 23 bugs are found. Assuming a Poisson distributi...
  - Gen 0: base=0.023 (X) -> ft=23.0 (V)
  - Gen 3: base=0.023 (X) -> ft=23.0 (V)

- **Q34** (0/4 -> 1/4): target=23717.0
  - Q: Determine the nearly minimum sample size (allowing at most 1 failure, i.e., acceptance number c = 1)...
  - Gen 3: base=25000.0 (X) -> ft=24000.0 (V)

- **Q49** (0/4 -> 1/4): target=180.0
  - Q: The lot acceptance criteria allows a maximum of three failures on 300 devices inspected (c = 3, n = ...
  - Gen 0: base=280.0 (X) -> ft=180.0 (V)

- **Q54** (1/4 -> 2/4): target=3.0
  - Q: A radar system consists of a critical power amplifier component that fails according to a Weibull di...
  - Gen 0: base=2.0 (X) -> ft=3.0 (V)
  - Gen 3: base=2.0 (X) -> ft=3.0 (V)

- **Q56** (1/4 -> 3/4): target=0.924
  - Q: A data center has 5 independent cooling units arranged in a k-out-of-n configuration where at least ...
  - Gen 0: base=0.9998 (X) -> ft=0.9177 (V)
  - Gen 1: base=1.0 (X) -> ft=0.9405 (V)

- **Q61** (0/4 -> 2/4): target=3169000.0
  - Q: A semiconductor device has a lifetime that follows a lognormal distribution with a shape parameter σ...
  - Gen 0: base=3000000.0 (X) -> ft=3169450.0 (V)
  - Gen 2: base=2664710.0 (X) -> ft=3028000.0 (V)

- **Q68** (3/4 -> 4/4): target=0.0067
  - Q: A power supply unit has a constant failure rate of λ = 0.004 failures per 1000 hours. The unit opera...
  - Gen 2: base=1.0 (X) -> ft=0.0067 (V)

- **Q70** (3/4 -> 4/4): target=0.5
  - Q: A communication system component has a reliability function (survival function) given by $R(t) = e^{...
  - Gen 1: base=-0.5 (X) -> ft=0.5 (V)

- **Q88** (3/4 -> 4/4): target=0.407
  - Q: A mechanical component has a lifetime that follows a Weibull distribution with shape parameter β = 2...
  - Gen 1: base=2.456 (X) -> ft=0.407 (V)

- **Q94** (2/4 -> 4/4): target=0.004105
  - Q: Consider a reliability test where five identical components are tested simultaneously. The failure t...
  - Gen 0: base=0.00447 (X) -> ft=0.0041 (V)
  - Gen 1: base=0.004472 (X) -> ft=0.0041 (V)

- **Q98** (3/4 -> 4/4): target=666.67
  - Q: A telecommunications server has a constant failure rate of 0.0015 failures per day. Assuming the exp...
  - Gen 0: base=6.0 (X) -> ft=666.67 (V)

- **Q104** (3/4 -> 4/4): target=5040.0
  - Q: A manufacturing facility produces circuit boards and conducts a life test on 10 identical boards. Th...
  - Gen 1: base=10000.0 (X) -> ft=5040.0 (V)

- **Q112** (0/4 -> 2/4): target=2.449
  - Q: A manufacturing facility experiences machine breakdowns that follow a Poisson process with a rate of...
  - Gen 1: base=6.0 (X) -> ft=2.449 (V)
  - Gen 3: base=6.0 (X) -> ft=2.449 (V)

- **Q116** (3/4 -> 4/4): target=0.566
  - Q: A telecommunications network uses a redundant server configuration to ensure high availability. The ...
  - Gen 3: base=0.0 (X) -> ft=0.568 (V)

- **Q118** (0/4 -> 1/4): target=145947.0
  - Q: A semiconductor manufacturer is evaluating the reliability of a new integrated circuit design. Field...
  - Gen 2: base=113000.0 (X) -> ft=141678.0 (V)

- **Q136** (0/4 -> 1/4): target=2916.0
  - Q: Industrial pumps used in chemical processing plants have lifetimes that follow a Weibull distributio...
  - Gen 2: base=3744.0 (X) -> ft=2936.0 (V)

- **Q168** (0/4 -> 1/4): target=0.0001315
  - Q: A semiconductor manufacturer is performing reliability qualification testing on a new integrated cir...
  - Gen 0: base=0.001869 (X) -> ft=0.0001327 (V)

#### Degraded questions (fine-tuned < base)

- **Q2** (4/4 -> 3/4): target=0.2841
  - Q: Three assembly plants produce the same parts. Plant A produces 25% of the volume and has a shipment ...
  - Gen 1: base=0.2841 (V) -> ft=0.0028439 (X)

- **Q5** (2/4 -> 1/4): target=0.6703
  - Q: An automatic misting system in an agricultural study follows a Poisson process with rate λ = 0.2 mis...
  - Gen 3: base=0.6703 (V) -> ft=-0.4 (X)

- **Q6** (4/4 -> 3/4): target=1132.0
  - Q: Assume that the lifetime follows a lognormal distribution, find the median life T₅₀ necessary for 5%...
  - Gen 1: base=1133.5 (V) -> ft=500.5 (X)

- **Q17** (4/4 -> 3/4): target=0.09121
  - Q: An electronic component acts to transform an input signal of level X into an output signal of level ...
  - Gen 0: base=0.0918 (V) -> ft=0.0718 (X)

- **Q21** (2/4 -> 1/4): target=2.0
  - Q: Consider the exponential distribution with MTTF = 50,000 hours. What is the failure rate in %/K hour...
  - Gen 1: base=2.0 (V) -> ft=None (X)
  - Gen 3: base=2.0 (V) -> ft=2e-06 (X)

- **Q25** (4/4 -> 3/4): target=0.866
  - Q: A baseball player has a current batting average of .250 (probability of a hit p = 0.250). In a game,...
  - Gen 3: base=0.866 (V) -> ft=None (X)

- **Q42** (4/4 -> 3/4): target=0.2424
  - Q: A memory chip lifetime for soft errors follows the exponential distribution with MTTF = 36 months. F...
  - Gen 3: base=0.243 (V) -> ft=None (X)

- **Q44** (2/4 -> 0/4): target=0.03285
  - Q: In an experiment to compare different treatments, the old method produced 4 rejects out of 20. In th...
  - Gen 1: base=0.0316 (V) -> ft=0.0307 (X)
  - Gen 2: base=0.0329 (V) -> ft=0.031 (X)

- **Q52** (1/4 -> 0/4): target=17.0
  - Q: A manufacturer wants to demonstrate an MTTF of at least 5,000 hours with 80% confidence using a 500-...
  - Gen 3: base=17.0 (V) -> ft=None (X)

- **Q60** (2/4 -> 0/4): target=0.368
  - Q: A system has a linearly increasing hazard rate given by h(t) = αt, where α is a positive constant an...
  - Gen 0: base=0.3679 (V) -> ft=-1.0 (X)
  - Gen 1: base=0.3679 (V) -> ft=-1.0 (X)

- **Q66** (2/4 -> 1/4): target=0.01
  - Q: A manufacturer monitors the defect rate in production batches. A new production method is implemente...
  - Gen 0: base=0.01 (V) -> ft=100.0 (X)
  - Gen 3: base=0.01 (V) -> ft=100.0 (X)

- **Q74** (2/4 -> 0/4): target=23790.0
  - Q: Electronic control units in automotive applications are found to have lifetimes that follow a Weibul...
  - Gen 1: base=23700.0 (V) -> ft=26250.0 (X)
  - Gen 3: base=23000.0 (V) -> ft=32445.0 (X)

- **Q82** (4/4 -> 3/4): target=7.7
  - Q: A component follows a lognormal distribution for its time-to-failure. Reliability testing shows that...
  - Gen 2: base=7.69 (V) -> ft=8.5 (X)

- **Q86** (4/4 -> 3/4): target=0.9818
  - Q: A satellite system consists of 3 redundant power modules configured in a 2-out-of-3 system (the syst...
  - Gen 3: base=0.9818 (V) -> ft=0.2031 (X)

- **Q125** (2/4 -> 1/4): target=11104.0
  - Q: A manufacturer produces industrial control units whose lifetimes follow a lognormal distribution wit...
  - Gen 0: base=10750.0 (V) -> ft=10336.0 (X)

- **Q126** (2/4 -> 1/4): target=2.08
  - Q: A manufacturer produces industrial control units whose lifetimes follow a lognormal distribution wit...
  - Gen 1: base=2.077 (V) -> ft=2.25 (X)

- **Q130** (4/4 -> 3/4): target=0.982
  - Q: A wind turbine blade monitoring system uses a 2-out-of-4 voting configuration for vibration sensors ...
  - Gen 0: base=0.9822 (V) -> ft=0.9145 (X)

- **Q137** (4/4 -> 3/4): target=0.8272
  - Q: Industrial pumps used in chemical processing plants have lifetimes that follow a Weibull distributio...
  - Gen 1: base=0.827 (V) -> ft=0.189 (X)

- **Q141** (3/4 -> 2/4): target=3.0
  - Q: A telecommunications satellite contains a critical oscillator component whose lifetime follows a Wei...
  - Gen 3: base=3.0 (V) -> ft=2.0 (X)

- **Q146** (3/4 -> 0/4): target=0.4493
  - Q: A data center monitors server failures across its infrastructure. Server crashes follow a Poisson pr...
  - Gen 1: base=0.4493 (V) -> ft=0.1353 (X)
  - Gen 2: base=0.4493 (V) -> ft=-0.8 (X)
  - Gen 3: base=0.4493 (V) -> ft=-0.8 (X)

- **Q149** (3/4 -> 2/4): target=6.0
  - Q: A telecommunications company is designing a redundant power supply system for a cell tower using ind...
  - Gen 1: base=6.0 (V) -> ft=None (X)

- **Q153** (4/4 -> 3/4): target=0.487
  - Q: A power distribution system for a manufacturing facility uses a network of independent circuit break...
  - Gen 3: base=0.4867 (V) -> ft=-0.72 (X)

- **Q167** (4/4 -> 3/4): target=17.0
  - Q: A software testing team is conducting defect analysis on a newly developed module containing 5,000 l...
  - Gen 2: base=17.0 (V) -> ft=0.085 (X)
