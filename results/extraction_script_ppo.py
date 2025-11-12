import re

log_text = """Logging to checkpoints
path
  Trial 1/50
    Current entropy coefficient: 0.700
    Exploration: Random action 0 (entropy-based)
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.000
    *** NEW BEST! ***
    Best config so far: [0.01 0.99 0.05]
    Best performance so far: 0.0
  Trial 2/50
    Current entropy coefficient: 0.683
    Exploitation: PPO action 27
    Selected action: 27
    Config: [0.1  0.99 0.6 ]
    Reward: 1.510
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.51
  Trial 3/50
    Current entropy coefficient: 0.666
    Exploration: Random action 37 (entropy-based)
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 1.195
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.51
  Trial 4/50
    Current entropy coefficient: 0.649
    Exploration: Random action 4 (entropy-based)
    Selected action: 4
    Config: [0.01 0.99 0.4 ]
    Reward: 1.698
    *** NEW BEST! ***
    Best config so far: [0.01 0.99 0.4 ]
    Best performance so far: 1.6985
  Trial 5/50
    Current entropy coefficient: 0.631
    Exploitation: PPO action 19
    Selected action: 19
    Config: [0.05 0.99 0.5 ]
    Reward: 1.743
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 6/50
    Current entropy coefficient: 0.614
    Exploration: Random action 21 (entropy-based)
    Selected action: 21
    Config: [0.1  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 7/50
    Current entropy coefficient: 0.597
    Exploitation: PPO action 26
    Selected action: 26
    Config: [0.1  0.99 0.5 ]
    Reward: 1.547
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 8/50
    Current entropy coefficient: 0.580
    Exploration: Random action 1 (entropy-based)
    Selected action: 1
    Config: [0.01 0.99 0.1 ]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 9/50
    Current entropy coefficient: 0.563
    Exploration: Random action 39 (entropy-based)
    Selected action: 39
    Config: [0.5  0.99 0.4 ]
    Reward: 0.864
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 10/50
    Current entropy coefficient: 0.546
    Exploration: Random action 3 (entropy-based)
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 1.715
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 11/50
    Current entropy coefficient: 0.529
    Exploration: Random action 25 (entropy-based)
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 1.694
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 12/50
    Current entropy coefficient: 0.511
    Exploration: Random action 38 (entropy-based)
    Selected action: 38
    Config: [0.5  0.99 0.3 ]
    Reward: 0.513
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 13/50
    Current entropy coefficient: 0.494
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.681
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 14/50
    Current entropy coefficient: 0.477
    Exploration: Random action 23 (entropy-based)
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 1.358
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 15/50
    Current entropy coefficient: 0.460
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.707
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 16/50
    Current entropy coefficient: 0.443
    Exploration: Random action 36 (entropy-based)
    Selected action: 36
    Config: [0.5  0.99 0.1 ]
    Reward: 0.351
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 17/50
    Current entropy coefficient: 0.426
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.696
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 18/50
    Current entropy coefficient: 0.409
    Exploration: Random action 6 (entropy-based)
    Selected action: 6
    Config: [0.01 0.99 0.6 ]
    Reward: 1.692
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 19/50
    Current entropy coefficient: 0.391
    Exploration: Random action 31 (entropy-based)
    Selected action: 31
    Config: [0.3  0.99 0.3 ]
    Reward: 1.202
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 20/50
    Current entropy coefficient: 0.374
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.696
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 21/50
    Current entropy coefficient: 0.357
    Exploration: Random action 37 (entropy-based)
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 0.878
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Current entropy coefficient: 0.340
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.700
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Current entropy coefficient: 0.323
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.714
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 24/50
    Current entropy coefficient: 0.306
    Exploitation: PPO action 31
    Selected action: 31
    Config: [0.3  0.99 0.3 ]
    Reward: 0.904
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 25/50
    Current entropy coefficient: 0.289
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.695
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 26/50
    Current entropy coefficient: 0.271
    Exploitation: PPO action 16
    Selected action: 16
    Config: [0.05 0.99 0.2 ]
    Reward: 1.695
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 27/50
    Current entropy coefficient: 0.254
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.716
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 28/50
    Current entropy coefficient: 0.237
    Exploration: Random action 3 (entropy-based)
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 1.686
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 29/50
    Current entropy coefficient: 0.220
    Exploration: Random action 3 (entropy-based)
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 1.740
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 30/50
    Current entropy coefficient: 0.203
    Exploration: Random action 34 (entropy-based)
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.362
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 31/50
    Current entropy coefficient: 0.186
    Exploitation: PPO action 33
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 1.371
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 32/50
    Current entropy coefficient: 0.169
    Exploration: Random action 20 (entropy-based)
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.711
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 33/50
    Current entropy coefficient: 0.151
    Exploitation: PPO action 16
    Selected action: 16
    Config: [0.05 0.99 0.2 ]
    Reward: 1.663
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 34/50
    Current entropy coefficient: 0.134
    Exploration: Random action 7 (entropy-based)
    Selected action: 7
    Config: [0.02 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
  Trial 35/50
    Current entropy coefficient: 0.117
    Exploitation: PPO action 25
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 1.239
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7425000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 36/50
    Current entropy coefficient: 0.100
    Exploration: Random action 33 (entropy-based)
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 1.746
    *** NEW BEST! ***
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 37/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 36
    Selected action: 36
    Config: [0.5  0.99 0.1 ]
    Reward: 0.021
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.032
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 39/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 15
    Selected action: 15
    Config: [0.05 0.99 0.1 ]
    Reward: 0.334
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
  Trial 40/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 24
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.672
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
  Trial 41/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.503
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
  Trial 42/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 4
    Selected action: 4
    Config: [0.01 0.99 0.4 ]
    Reward: 1.712
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 43/50
    Current entropy coefficient: 0.100
    Exploration: Random action 31 (entropy-based)
    Selected action: 31
    Config: [0.3  0.99 0.3 ]
    Reward: 1.188
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 44/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.024
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.211
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 46/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.743
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
  Trial 47/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.713
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 48/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 11
    Selected action: 11
    Config: [0.02 0.99 0.4 ]
    Reward: 1.685
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Current entropy coefficient: 0.100
    Exploration: Random action 35 (entropy-based)
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
  Trial 50/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.709
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 1.7465
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.3  0.99 0.5 ]
Best performance: 1.746
Total trials completed: 50
Logging to checkpoints
path
  Trial 1/50
    Current entropy coefficient: 0.700
    Exploitation: PPO action 22
    Selected action: 22
    Config: [0.1  0.99 0.1 ]
    Reward: 1.001
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.1 ]
    Best performance so far: 1.0010000000000001
  Trial 2/50
    Current entropy coefficient: 0.683
    Exploitation: PPO action 27
    Selected action: 27
    Config: [0.1  0.99 0.6 ]
    Reward: 1.737
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 3/50
    Current entropy coefficient: 0.666
    Exploration: Random action 7 (entropy-based)
    Selected action: 7
    Config: [0.02 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 4/50
    Current entropy coefficient: 0.649
    Exploitation: PPO action 19
    Selected action: 19
    Config: [0.05 0.99 0.5 ]
    Reward: 1.661
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 5/50
    Current entropy coefficient: 0.631
    Exploration: Random action 7 (entropy-based)
    Selected action: 7
    Config: [0.02 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
    Meta-environment terminated early, but continuing HPO...
  Trial 6/50
    Current entropy coefficient: 0.614
    Exploitation: PPO action 26
    Selected action: 26
    Config: [0.1  0.99 0.5 ]
    Reward: 1.667
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 7/50
    Current entropy coefficient: 0.597
    Exploration: Random action 24 (entropy-based)
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.679
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 8/50
    Current entropy coefficient: 0.580
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.669
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 9/50
    Current entropy coefficient: 0.563
    Exploration: Random action 38 (entropy-based)
    Selected action: 38
    Config: [0.5  0.99 0.3 ]
    Reward: 0.807
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 10/50
    Current entropy coefficient: 0.546
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.725
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
    Meta-environment terminated early, but continuing HPO...
  Trial 11/50
    Current entropy coefficient: 0.529
    Exploration: Random action 5 (entropy-based)
    Selected action: 5
    Config: [0.01 0.99 0.5 ]
    Reward: 1.715
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 12/50
    Current entropy coefficient: 0.511
    Exploration: Random action 3 (entropy-based)
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 1.714
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 13/50
    Current entropy coefficient: 0.494
    Exploration: Random action 34 (entropy-based)
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.375
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 14/50
    Current entropy coefficient: 0.477
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.727
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 15/50
    Current entropy coefficient: 0.460
    Exploration: Random action 5 (entropy-based)
    Selected action: 5
    Config: [0.01 0.99 0.5 ]
    Reward: 1.732
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
    Meta-environment terminated early, but continuing HPO...
  Trial 16/50
    Current entropy coefficient: 0.443
    Exploration: Random action 35 (entropy-based)
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 17/50
    Current entropy coefficient: 0.426
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.709
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 18/50
    Current entropy coefficient: 0.409
    Exploration: Random action 39 (entropy-based)
    Selected action: 39
    Config: [0.5  0.99 0.4 ]
    Reward: 0.481
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 19/50
    Current entropy coefficient: 0.391
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.545
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 20/50
    Current entropy coefficient: 0.374
    Exploration: Random action 29 (entropy-based)
    Selected action: 29
    Config: [0.3  0.99 0.1 ]
    Reward: 0.350
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 21/50
    Current entropy coefficient: 0.357
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.733
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Current entropy coefficient: 0.340
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.699
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Current entropy coefficient: 0.323
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.666
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
    Meta-environment terminated early, but continuing HPO...
  Trial 24/50
    Current entropy coefficient: 0.306
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.869
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
  Trial 25/50
    Current entropy coefficient: 0.289
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.722
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.737
    Meta-environment terminated early, but continuing HPO...
  Trial 26/50
    Current entropy coefficient: 0.271
    Exploration: Random action 19 (entropy-based)
    Selected action: 19
    Config: [0.05 0.99 0.5 ]
    Reward: 1.744
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 27/50
    Current entropy coefficient: 0.254
    Exploitation: PPO action 37
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 1.026
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
  Trial 28/50
    Current entropy coefficient: 0.237
    Exploration: Random action 18 (entropy-based)
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.667
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 29/50
    Current entropy coefficient: 0.220
    Exploitation: PPO action 33
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 1.192
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
  Trial 30/50
    Current entropy coefficient: 0.203
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.688
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Current entropy coefficient: 0.186
    Exploitation: PPO action 0
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
  Trial 32/50
    Current entropy coefficient: 0.169
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.735
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 33/50
    Current entropy coefficient: 0.151
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.568
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 34/50
    Current entropy coefficient: 0.134
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.708
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 35/50
    Current entropy coefficient: 0.117
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.102
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 36/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.720
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 37/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.695
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 0.876
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 39/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.722
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 40/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.674
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
  Trial 41/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 0.713
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 42/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.396
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 43/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 30
    Selected action: 30
    Config: [0.3  0.99 0.2 ]
    Reward: 1.337
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
  Trial 44/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 23
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 1.623
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
  Trial 45/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 4
    Selected action: 4
    Config: [0.01 0.99 0.4 ]
    Reward: 1.698
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
  Trial 46/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 12
    Selected action: 12
    Config: [0.02 0.99 0.5 ]
    Reward: 1.712
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
  Trial 47/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.728
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 48/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 25
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 1.411
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.725
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    Meta-environment terminated early, but continuing HPO...
  Trial 50/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.701
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7440000000000002
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.05 0.99 0.5 ]
Best performance: 1.744
Total trials completed: 50
Logging to checkpoints
path
  Trial 1/50
    Current entropy coefficient: 0.700
    Exploration: Random action 8 (entropy-based)
    Selected action: 8
    Config: [0.02 0.99 0.1 ]
    Reward: 0.000
    *** NEW BEST! ***
    Best config so far: [0.02 0.99 0.1 ]
    Best performance so far: 0.0
  Trial 2/50
    Current entropy coefficient: 0.683
    Exploitation: PPO action 27
    Selected action: 27
    Config: [0.1  0.99 0.6 ]
    Reward: 1.671
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.671
  Trial 3/50
    Current entropy coefficient: 0.666
    Exploration: Random action 30 (entropy-based)
    Selected action: 30
    Config: [0.3  0.99 0.2 ]
    Reward: 1.219
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 1.671
  Trial 4/50
    Current entropy coefficient: 0.649
    Exploitation: PPO action 13
    Selected action: 13
    Config: [0.02 0.99 0.6 ]
    Reward: 1.741
    *** NEW BEST! ***
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 1.741
  Trial 5/50
    Current entropy coefficient: 0.631
    Exploration: Random action 36 (entropy-based)
    Selected action: 36
    Config: [0.5  0.99 0.1 ]
    Reward: 0.555
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 1.741
  Trial 6/50
    Current entropy coefficient: 0.614
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.367
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 1.741
  Trial 7/50
    Current entropy coefficient: 0.597
    Exploration: Random action 32 (entropy-based)
    Selected action: 32
    Config: [0.3  0.99 0.4 ]
    Reward: 1.243
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 1.741
  Trial 8/50
    Current entropy coefficient: 0.580
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.683
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 1.741
  Trial 9/50
    Current entropy coefficient: 0.563
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.747
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 10/50
    Current entropy coefficient: 0.546
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.709
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 11/50
    Current entropy coefficient: 0.529
    Exploration: Random action 4 (entropy-based)
    Selected action: 4
    Config: [0.01 0.99 0.4 ]
    Reward: 1.683
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 12/50
    Current entropy coefficient: 0.511
    Exploration: Random action 0 (entropy-based)
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 13/50
    Current entropy coefficient: 0.494
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.703
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 14/50
    Current entropy coefficient: 0.477
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.692
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 15/50
    Current entropy coefficient: 0.460
    Exploration: Random action 24 (entropy-based)
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.692
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 16/50
    Current entropy coefficient: 0.443
    Exploration: Random action 29 (entropy-based)
    Selected action: 29
    Config: [0.3  0.99 0.1 ]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 17/50
    Current entropy coefficient: 0.426
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.705
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 18/50
    Current entropy coefficient: 0.409
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.675
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 19/50
    Current entropy coefficient: 0.391
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.714
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 20/50
    Current entropy coefficient: 0.374
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.666
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 21/50
    Current entropy coefficient: 0.357
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.673
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Current entropy coefficient: 0.340
    Exploration: Random action 9 (entropy-based)
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.700
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Current entropy coefficient: 0.323
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.548
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 24/50
    Current entropy coefficient: 0.306
    Exploitation: PPO action 31
    Selected action: 31
    Config: [0.3  0.99 0.3 ]
    Reward: 1.224
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 25/50
    Current entropy coefficient: 0.289
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.342
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 26/50
    Current entropy coefficient: 0.271
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.711
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 27/50
    Current entropy coefficient: 0.254
    Exploitation: PPO action 23
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 1.497
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 28/50
    Current entropy coefficient: 0.237
    Exploitation: PPO action 26
    Selected action: 26
    Config: [0.1  0.99 0.5 ]
    Reward: 1.677
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 29/50
    Current entropy coefficient: 0.220
    Exploration: Random action 1 (entropy-based)
    Selected action: 1
    Config: [0.01 0.99 0.1 ]
    Reward: 0.560
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 30/50
    Current entropy coefficient: 0.203
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.674
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Current entropy coefficient: 0.186
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.699
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 32/50
    Current entropy coefficient: 0.169
    Exploitation: PPO action 26
    Selected action: 26
    Config: [0.1  0.99 0.5 ]
    Reward: 1.722
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 33/50
    Current entropy coefficient: 0.151
    Exploitation: PPO action 33
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 0.894
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 34/50
    Current entropy coefficient: 0.134
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.718
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 35/50
    Current entropy coefficient: 0.117
    Exploitation: PPO action 25
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 1.680
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 36/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 33
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 1.217
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 37/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.746
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 25
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 1.711
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 39/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 37
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 0.677
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 40/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.733
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 41/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 33
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 0.530
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 42/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.496
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 43/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 23
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 1.527
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 44/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 24
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.735
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Current entropy coefficient: 0.100
    Exploration: Random action 35 (entropy-based)
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 46/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 2
    Selected action: 2
    Config: [0.01 0.99 0.2 ]
    Reward: 1.702
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 47/50
    Current entropy coefficient: 0.100
    Exploration: Random action 21 (entropy-based)
    Selected action: 21
    Config: [0.1  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 48/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 12
    Selected action: 12
    Config: [0.02 0.99 0.5 ]
    Reward: 1.731
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 11
    Selected action: 11
    Config: [0.02 0.99 0.4 ]
    Reward: 1.698
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
  Trial 50/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.693
    Best config so far: [0.05 0.99 0.4 ]
    Best performance so far: 1.7474999999999998
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.05 0.99 0.4 ]
Best performance: 1.747
Total trials completed: 50
Logging to checkpoints
path
  Trial 1/50
    Current entropy coefficient: 0.700
    Exploration: Random action 3 (entropy-based)
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 1.704
    *** NEW BEST! ***
    Best config so far: [0.01 0.99 0.3 ]
    Best performance so far: 1.7035
  Trial 2/50
    Current entropy coefficient: 0.683
    Exploitation: PPO action 27
    Selected action: 27
    Config: [0.1  0.99 0.6 ]
    Reward: 1.509
    Best config so far: [0.01 0.99 0.3 ]
    Best performance so far: 1.7035
  Trial 3/50
    Current entropy coefficient: 0.666
    Exploitation: PPO action 28
    Selected action: 28
    Config: [0.3  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.01 0.99 0.3 ]
    Best performance so far: 1.7035
  Trial 4/50
    Current entropy coefficient: 0.649
    Exploitation: PPO action 19
    Selected action: 19
    Config: [0.05 0.99 0.5 ]
    Reward: 1.722
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7215
  Trial 5/50
    Current entropy coefficient: 0.631
    Exploration: Random action 38 (entropy-based)
    Selected action: 38
    Config: [0.5  0.99 0.3 ]
    Reward: 1.200
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7215
  Trial 6/50
    Current entropy coefficient: 0.614
    Exploration: Random action 11 (entropy-based)
    Selected action: 11
    Config: [0.02 0.99 0.4 ]
    Reward: 1.716
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7215
  Trial 7/50
    Current entropy coefficient: 0.597
    Exploration: Random action 34 (entropy-based)
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 0.888
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7215
  Trial 8/50
    Current entropy coefficient: 0.580
    Exploration: Random action 35 (entropy-based)
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7215
  Trial 9/50
    Current entropy coefficient: 0.563
    Exploitation: PPO action 29
    Selected action: 29
    Config: [0.3  0.99 0.1 ]
    Reward: 0.484
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7215
  Trial 10/50
    Current entropy coefficient: 0.546
    Exploitation: PPO action 24
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.163
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.7215
  Trial 11/50
    Current entropy coefficient: 0.529
    Exploration: Random action 11 (entropy-based)
    Selected action: 11
    Config: [0.02 0.99 0.4 ]
    Reward: 1.724
    *** NEW BEST! ***
    Best config so far: [0.02 0.99 0.4 ]
    Best performance so far: 1.7235
    Meta-environment terminated early, but continuing HPO...
  Trial 12/50
    Current entropy coefficient: 0.511
    Exploration: Random action 0 (entropy-based)
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.02 0.99 0.4 ]
    Best performance so far: 1.7235
  Trial 13/50
    Current entropy coefficient: 0.494
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.512
    Best config so far: [0.02 0.99 0.4 ]
    Best performance so far: 1.7235
  Trial 14/50
    Current entropy coefficient: 0.477
    Exploration: Random action 8 (entropy-based)
    Selected action: 8
    Config: [0.02 0.99 0.1 ]
    Reward: 0.337
    Best config so far: [0.02 0.99 0.4 ]
    Best performance so far: 1.7235
  Trial 15/50
    Current entropy coefficient: 0.460
    Exploitation: PPO action 0
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.02 0.99 0.4 ]
    Best performance so far: 1.7235
    Meta-environment terminated early, but continuing HPO...
  Trial 16/50
    Current entropy coefficient: 0.443
    Exploitation: PPO action 24
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.210
    Best config so far: [0.02 0.99 0.4 ]
    Best performance so far: 1.7235
    Meta-environment terminated early, but continuing HPO...
  Trial 17/50
    Current entropy coefficient: 0.426
    Exploitation: PPO action 29
    Selected action: 29
    Config: [0.3  0.99 0.1 ]
    Reward: 0.182
    Best config so far: [0.02 0.99 0.4 ]
    Best performance so far: 1.7235
    Meta-environment terminated early, but continuing HPO...
  Trial 18/50
    Current entropy coefficient: 0.409
    Exploitation: PPO action 23
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 1.745
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
  Trial 19/50
    Current entropy coefficient: 0.391
    Exploitation: PPO action 23
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 1.203
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 20/50
    Current entropy coefficient: 0.374
    Exploration: Random action 0 (entropy-based)
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.170
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 21/50
    Current entropy coefficient: 0.357
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.714
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Current entropy coefficient: 0.340
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.669
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
  Trial 23/50
    Current entropy coefficient: 0.323
    Exploitation: PPO action 29
    Selected action: 29
    Config: [0.3  0.99 0.1 ]
    Reward: 0.735
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 24/50
    Current entropy coefficient: 0.306
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.729
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 25/50
    Current entropy coefficient: 0.289
    Exploitation: PPO action 24
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.570
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 26/50
    Current entropy coefficient: 0.271
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.879
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
  Trial 27/50
    Current entropy coefficient: 0.254
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.694
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
  Trial 28/50
    Current entropy coefficient: 0.237
    Exploitation: PPO action 3
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 1.726
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 29/50
    Current entropy coefficient: 0.220
    Exploitation: PPO action 24
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.544
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 30/50
    Current entropy coefficient: 0.203
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.208
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Current entropy coefficient: 0.186
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.562
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 32/50
    Current entropy coefficient: 0.169
    Exploitation: PPO action 17
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 1.724
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
  Trial 33/50
    Current entropy coefficient: 0.151
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.907
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 34/50
    Current entropy coefficient: 0.134
    Exploitation: PPO action 36
    Selected action: 36
    Config: [0.5  0.99 0.1 ]
    Reward: 0.506
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
  Trial 35/50
    Current entropy coefficient: 0.117
    Exploitation: PPO action 30
    Selected action: 30
    Config: [0.3  0.99 0.2 ]
    Reward: 0.513
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
  Trial 36/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 12
    Selected action: 12
    Config: [0.02 0.99 0.5 ]
    Reward: 1.652
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
  Trial 37/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 36
    Selected action: 36
    Config: [0.5  0.99 0.1 ]
    Reward: 0.350
    Best config so far: [0.1  0.99 0.2 ]
    Best performance so far: 1.7449999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.762
    *** NEW BEST! ***
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 39/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 23
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 1.707
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 40/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 12
    Selected action: 12
    Config: [0.02 0.99 0.5 ]
    Reward: 1.709
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 41/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 12
    Selected action: 12
    Config: [0.02 0.99 0.5 ]
    Reward: 1.679
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 42/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 25
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 1.595
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
  Trial 43/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.687
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
  Trial 44/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.695
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.337
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 46/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 12
    Selected action: 12
    Config: [0.02 0.99 0.5 ]
    Reward: 1.727
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 47/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 4
    Selected action: 4
    Config: [0.01 0.99 0.4 ]
    Reward: 1.716
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
  Trial 48/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.748
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.850
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    Meta-environment terminated early, but continuing HPO...
  Trial 50/50
    Current entropy coefficient: 0.100
    Exploration: Random action 27 (entropy-based)
    Selected action: 27
    Config: [0.1  0.99 0.6 ]
    Reward: 1.551
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.762
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.02 0.99 0.2 ]
Best performance: 1.762
Total trials completed: 50
Logging to checkpoints
path
  Trial 1/50
    Current entropy coefficient: 0.700
    Exploitation: PPO action 22
    Selected action: 22
    Config: [0.1  0.99 0.1 ]
    Reward: 0.338
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.1 ]
    Best performance so far: 0.33849999999999997
  Trial 2/50
    Current entropy coefficient: 0.683
    Exploration: Random action 24 (entropy-based)
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.544
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.3 ]
    Best performance so far: 1.544
  Trial 3/50
    Current entropy coefficient: 0.666
    Exploration: Random action 14 (entropy-based)
    Selected action: 14
    Config: [0.05 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.1  0.99 0.3 ]
    Best performance so far: 1.544
  Trial 4/50
    Current entropy coefficient: 0.649
    Exploitation: PPO action 19
    Selected action: 19
    Config: [0.05 0.99 0.5 ]
    Reward: 1.666
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.6664999999999999
  Trial 5/50
    Current entropy coefficient: 0.631
    Exploitation: PPO action 19
    Selected action: 19
    Config: [0.05 0.99 0.5 ]
    Reward: 1.671
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.671
    Meta-environment terminated early, but continuing HPO...
  Trial 6/50
    Current entropy coefficient: 0.614
    Exploration: Random action 35 (entropy-based)
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.671
  Trial 7/50
    Current entropy coefficient: 0.597
    Exploitation: PPO action 25
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 1.538
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.671
  Trial 8/50
    Current entropy coefficient: 0.580
    Exploration: Random action 37 (entropy-based)
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 0.513
    Best config so far: [0.05 0.99 0.5 ]
    Best performance so far: 1.671
  Trial 9/50
    Current entropy coefficient: 0.563
    Exploration: Random action 9 (entropy-based)
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.735
    *** NEW BEST! ***
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
  Trial 10/50
    Current entropy coefficient: 0.546
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.718
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
  Trial 11/50
    Current entropy coefficient: 0.529
    Exploration: Random action 25 (entropy-based)
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 1.713
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
    Meta-environment terminated early, but continuing HPO...
  Trial 12/50
    Current entropy coefficient: 0.511
    Exploration: Random action 6 (entropy-based)
    Selected action: 6
    Config: [0.01 0.99 0.6 ]
    Reward: 1.704
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
  Trial 13/50
    Current entropy coefficient: 0.494
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.674
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
    Meta-environment terminated early, but continuing HPO...
  Trial 14/50
    Current entropy coefficient: 0.477
    Exploration: Random action 4 (entropy-based)
    Selected action: 4
    Config: [0.01 0.99 0.4 ]
    Reward: 1.709
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
  Trial 15/50
    Current entropy coefficient: 0.460
    Exploration: Random action 20 (entropy-based)
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.681
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
    Meta-environment terminated early, but continuing HPO...
  Trial 16/50
    Current entropy coefficient: 0.443
    Exploration: Random action 30 (entropy-based)
    Selected action: 30
    Config: [0.3  0.99 0.2 ]
    Reward: 1.196
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
  Trial 17/50
    Current entropy coefficient: 0.426
    Exploitation: PPO action 8
    Selected action: 8
    Config: [0.02 0.99 0.1 ]
    Reward: 0.000
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
  Trial 18/50
    Current entropy coefficient: 0.409
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.714
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
    Meta-environment terminated early, but continuing HPO...
  Trial 19/50
    Current entropy coefficient: 0.391
    Exploitation: PPO action 7
    Selected action: 7
    Config: [0.02 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
  Trial 20/50
    Current entropy coefficient: 0.374
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.329
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
    Meta-environment terminated early, but continuing HPO...
  Trial 21/50
    Current entropy coefficient: 0.357
    Exploration: Random action 14 (entropy-based)
    Selected action: 14
    Config: [0.05 0.99 0.05]
    Reward: 0.000
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.735
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Current entropy coefficient: 0.340
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.752
    *** NEW BEST! ***
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Current entropy coefficient: 0.323
    Exploitation: PPO action 21
    Selected action: 21
    Config: [0.1  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 24/50
    Current entropy coefficient: 0.306
    Exploitation: PPO action 33
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 1.226
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 25/50
    Current entropy coefficient: 0.289
    Exploitation: PPO action 23
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 1.343
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 26/50
    Current entropy coefficient: 0.271
    Exploitation: PPO action 24
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 1.688
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 27/50
    Current entropy coefficient: 0.254
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.733
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 28/50
    Current entropy coefficient: 0.237
    Exploitation: PPO action 37
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 0.543
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 29/50
    Current entropy coefficient: 0.220
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.695
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 30/50
    Current entropy coefficient: 0.203
    Exploration: Random action 22 (entropy-based)
    Selected action: 22
    Config: [0.1  0.99 0.1 ]
    Reward: 0.000
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Current entropy coefficient: 0.186
    Exploitation: PPO action 30
    Selected action: 30
    Config: [0.3  0.99 0.2 ]
    Reward: 1.299
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 32/50
    Current entropy coefficient: 0.169
    Exploitation: PPO action 35
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.000
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 33/50
    Current entropy coefficient: 0.151
    Exploitation: PPO action 9
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 1.687
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 34/50
    Current entropy coefficient: 0.134
    Exploitation: PPO action 11
    Selected action: 11
    Config: [0.02 0.99 0.4 ]
    Reward: 1.725
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 35/50
    Current entropy coefficient: 0.117
    Exploitation: PPO action 37
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 0.702
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 36/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 12
    Selected action: 12
    Config: [0.02 0.99 0.5 ]
    Reward: 1.711
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 37/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 2
    Selected action: 2
    Config: [0.01 0.99 0.2 ]
    Reward: 1.677
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 38/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.692
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 39/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 12
    Selected action: 12
    Config: [0.02 0.99 0.5 ]
    Reward: 1.700
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 40/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.680
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 41/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.744
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 42/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 18
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 1.719
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 43/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 11
    Selected action: 11
    Config: [0.02 0.99 0.4 ]
    Reward: 1.683
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 44/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.698
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 33
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 1.224
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 46/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 34
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 1.364
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 47/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 41
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 1.209
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
  Trial 48/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 10
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 1.689
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Current entropy coefficient: 0.100
    Exploitation: PPO action 20
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 1.682
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    Meta-environment terminated early, but continuing HPO...
  Trial 50/50
    Current entropy coefficient: 0.100
    Exploration: Random action 36 (entropy-based)
    Selected action: 36
    Config: [0.5  0.99 0.1 ]
    Reward: 1.085
    Best config so far: [0.02 0.99 0.2 ]
    Best performance so far: 1.752
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.02 0.99 0.2 ]
Best performance: 1.752
Total trials completed: 50"""

pattern = re.compile(r"Reward: ([\d.]+)")

# Find all matches (all trial reward values) in the log text
extracted_rewards_str = pattern.findall(log_text)

# Convert the captured strings to floats and save them in the initial flat list
trial_rewards_flat = [float(re.sub(r'[^0-9.]', '', r)) for r in extracted_rewards_str] # Use re.sub to clean up any extra whitespace/newline if necessary, though the pattern should capture cleanly.

# --- Restructure the data into 5 subarrays of length 10 ---
trials_per_run = 50
restructured_array = []
num_runs = len(trial_rewards_flat) // trials_per_run # Should be 50 / 10 = 5

for i in range(num_runs):
    # Calculate the start and end index for the current run (sub-array)
    start_index = i * trials_per_run
    end_index = (i + 1) * trials_per_run
    
    # Slice the flat list to get the 10 trials for this run
    run_trials = trial_rewards_flat[start_index:end_index]
    
    # Append the list of 10 trials to the main restructured array
    restructured_array.append(run_trials)

# --- Output and Verification ---

print("--- Extracted and Restructured Trial Rewards ---")
print(f"Total Runs (Sub-arrays): {len(restructured_array)}")
print(f"Length of each Run Array: {len(restructured_array[0]) if restructured_array else 0}")

# Print the restructured array with some formatting for clarity
print("\nRestructured Array (5 Runs x 10 Trials):")
for i, run in enumerate(restructured_array):
    # Format the floats to 4 decimal places for cleaner output
    formatted_run = [f'{r:.4f}' for r in run]
    print(f"Run {i+1}: {formatted_run}")

# The final variable containing the desired structure is 'restructured_array'
print("\nPython Variable 'restructured_array' content (raw format):")
print(restructured_array)
