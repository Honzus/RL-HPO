import re

log_text = """Run 1
Trial 1/50
    Selected action: 38
    Config: [0.5  0.99 0.3 ]
    Reward: 0.758
    *** NEW BEST! ***
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.758
  Trial 2/50
    Selected action: 21
    Config: [0.1  0.99 0.05]
    Reward: 0.773
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.05]
    Best performance so far: 0.7729999999999999
  Trial 3/50
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 0.787
    *** NEW BEST! ***
    Best config so far: [0.01 0.99 0.3 ]
    Best performance so far: 0.787
  Trial 4/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.773
    Best config so far: [0.01 0.99 0.3 ]
    Best performance so far: 0.787
  Trial 5/50
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 0.794
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 6/50
    Selected action: 19
    Config: [0.05 0.99 0.5 ]
    Reward: 0.787
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 7/50
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 8/50
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.785
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 9/50
    Selected action: 7
    Config: [0.02 0.99 0.05]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 10/50
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.786
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 11/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 12/50
    Selected action: 38
    Config: [0.5  0.99 0.3 ]
    Reward: 0.778
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 13/50
    Selected action: 36
    Config: [0.5  0.99 0.1 ]
    Reward: 0.779
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 14/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.783
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 15/50
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 0.767
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 16/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.769
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 17/50
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 18/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.775
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 19/50
    Selected action: 7
    Config: [0.02 0.99 0.05]
    Reward: 0.769
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 20/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.785
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 21/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.787
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.775
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.775
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 24/50
    Selected action: 14
    Config: [0.05 0.99 0.05]
    Reward: 0.772
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 25/50
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 0.779
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 26/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.780
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 27/50
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 0.786
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 28/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.791
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 29/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.779
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 30/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.781
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 32/50
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
  Trial 33/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.772
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 34/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.768
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 35/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 36/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.786
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 37/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.787
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 39/50
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 0.785
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 40/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.778
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 41/50
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.772
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 42/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.772
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 43/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.769
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 44/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.770
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.785
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 46/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.773
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 47/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.772
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 48/50
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.791
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.788
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 50/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.790
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7939999999999999
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.05 0.99 0.6 ]
Best performance: 0.794
Total trials completed: 50

Run 2
Trial 1/50
    Selected action: 15
    Config: [0.05 0.99 0.1 ]
    Reward: 0.771
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.1 ]
    Best performance so far: 0.771
  Trial 2/50
    Selected action: 21
    Config: [0.1  0.99 0.05]
    Reward: 0.779
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.05]
    Best performance so far: 0.7790000000000001
  Trial 3/50
    Selected action: 4
    Config: [0.01 0.99 0.4 ]
    Reward: 0.775
    Best config so far: [0.1  0.99 0.05]
    Best performance so far: 0.7790000000000001
  Trial 4/50
    Selected action: 27
    Config: [0.1  0.99 0.6 ]
    Reward: 0.779
    Best config so far: [0.1  0.99 0.05]
    Best performance so far: 0.7790000000000001
  Trial 5/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.773
    Best config so far: [0.1  0.99 0.05]
    Best performance so far: 0.7790000000000001
  Trial 6/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.769
    Best config so far: [0.1  0.99 0.05]
    Best performance so far: 0.7790000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 7/50
    Selected action: 5
    Config: [0.01 0.99 0.5 ]
    Reward: 0.780
    *** NEW BEST! ***
    Best config so far: [0.01 0.99 0.5 ]
    Best performance so far: 0.78
  Trial 8/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.784
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.3 ]
    Best performance so far: 0.784
    Meta-environment terminated early, but continuing HPO...
  Trial 9/50
    Selected action: 13
    Config: [0.02 0.99 0.6 ]
    Reward: 0.783
    Best config so far: [0.1  0.99 0.3 ]
    Best performance so far: 0.784
  Trial 10/50
    Selected action: 16
    Config: [0.05 0.99 0.2 ]
    Reward: 0.791
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 11/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.780
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 12/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.778
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 13/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.788
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 14/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 15/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 16/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.781
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 17/50
    Selected action: 6
    Config: [0.01 0.99 0.6 ]
    Reward: 0.776
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 18/50
    Selected action: 15
    Config: [0.05 0.99 0.1 ]
    Reward: 0.767
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 19/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.778
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 20/50
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.775
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 21/50
    Selected action: 6
    Config: [0.01 0.99 0.6 ]
    Reward: 0.777
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.788
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.768
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 24/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.772
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 25/50
    Selected action: 15
    Config: [0.05 0.99 0.1 ]
    Reward: 0.761
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 26/50
    Selected action: 6
    Config: [0.01 0.99 0.6 ]
    Reward: 0.789
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 27/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 28/50
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 0.776
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 29/50
    Selected action: 29
    Config: [0.3  0.99 0.1 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 30/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.783
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.771
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 32/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.773
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 33/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.769
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 34/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 35/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.777
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 36/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.785
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 37/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Selected action: 31
    Config: [0.3  0.99 0.3 ]
    Reward: 0.778
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 39/50
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 0.776
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 40/50
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 0.770
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 41/50
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 0.786
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 42/50
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 0.780
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
  Trial 43/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.776
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 44/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.779
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.778
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 46/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.769
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 47/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.787
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 48/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.773
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    Meta-environment terminated early, but continuing HPO...
  Trial 50/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.781
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.7910000000000001
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.05 0.99 0.2 ]
Best performance: 0.791
Total trials completed: 50

Run 3
Trial 1/50
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 0.788
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7879999999999999
  Trial 2/50
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.769
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.7879999999999999
  Trial 3/50
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 0.788
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.788
  Trial 4/50
    Selected action: 1
    Config: [0.01 0.99 0.1 ]
    Reward: 0.789
    *** NEW BEST! ***
    Best config so far: [0.01 0.99 0.1 ]
    Best performance so far: 0.7889999999999999
  Trial 5/50
    Selected action: 31
    Config: [0.3  0.99 0.3 ]
    Reward: 0.785
    Best config so far: [0.01 0.99 0.1 ]
    Best performance so far: 0.7889999999999999
  Trial 6/50
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 0.783
    Best config so far: [0.01 0.99 0.1 ]
    Best performance so far: 0.7889999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 7/50
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 0.759
    Best config so far: [0.01 0.99 0.1 ]
    Best performance so far: 0.7889999999999999
  Trial 8/50
    Selected action: 18
    Config: [0.05 0.99 0.4 ]
    Reward: 0.773
    Best config so far: [0.01 0.99 0.1 ]
    Best performance so far: 0.7889999999999999
  Trial 9/50
    Selected action: 27
    Config: [0.1  0.99 0.6 ]
    Reward: 0.791
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 0.791
  Trial 10/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.765
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 0.791
  Trial 11/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.785
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 0.791
  Trial 12/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.784
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 13/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 14/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.778
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 15/50
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.1  0.99 0.6 ]
    Best performance so far: 0.791
  Trial 16/50
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.798
    *** NEW BEST! ***
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
  Trial 17/50
    Selected action: 16
    Config: [0.05 0.99 0.2 ]
    Reward: 0.790
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
  Trial 18/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.779
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 19/50
    Selected action: 30
    Config: [0.3  0.99 0.2 ]
    Reward: 0.776
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
  Trial 20/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.777
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 21/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.777
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.775
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Selected action: 8
    Config: [0.02 0.99 0.1 ]
    Reward: 0.768
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
  Trial 24/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.783
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 25/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.765
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 26/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.780
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 27/50
    Selected action: 10
    Config: [0.02 0.99 0.3 ]
    Reward: 0.781
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
  Trial 28/50
    Selected action: 41
    Config: [0.5  0.99 0.6 ]
    Reward: 0.782
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
  Trial 29/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.779
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 30/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.784
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.787
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 32/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.795
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 33/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 34/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.782
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 35/50
    Selected action: 26
    Config: [0.1  0.99 0.5 ]
    Reward: 0.777
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
  Trial 36/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.775
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 37/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.771
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 39/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.783
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 40/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.784
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 41/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.777
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 42/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.781
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 43/50
    Selected action: 21
    Config: [0.1  0.99 0.05]
    Reward: 0.774
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
  Trial 44/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.789
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.770
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 46/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.781
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 47/50
    Selected action: 0
    Config: [0.01 0.99 0.05]
    Reward: 0.770
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 48/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.791
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.783
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    Meta-environment terminated early, but continuing HPO...
  Trial 50/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.786
    Best config so far: [0.5  0.99 0.05]
    Best performance so far: 0.798
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.5  0.99 0.05]
Best performance: 0.798
Total trials completed: 50

Run 4
Trial 1/50
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 0.776
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.776
  Trial 2/50
    Selected action: 21
    Config: [0.1  0.99 0.05]
    Reward: 0.775
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.776
  Trial 3/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.772
    Best config so far: [0.05 0.99 0.6 ]
    Best performance so far: 0.776
  Trial 4/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.780
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.3 ]
    Best performance so far: 0.78
  Trial 5/50
    Selected action: 33
    Config: [0.3  0.99 0.5 ]
    Reward: 0.794
    *** NEW BEST! ***
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 0.7939999999999999
  Trial 6/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.769
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 7/50
    Selected action: 3
    Config: [0.01 0.99 0.3 ]
    Reward: 0.788
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 0.7939999999999999
  Trial 8/50
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 0.776
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 0.7939999999999999
  Trial 9/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.763
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 10/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.778
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 11/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.781
    Best config so far: [0.3  0.99 0.5 ]
    Best performance so far: 0.7939999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 12/50
    Selected action: 25
    Config: [0.1  0.99 0.4 ]
    Reward: 0.797
    *** NEW BEST! ***
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 13/50
    Selected action: 29
    Config: [0.3  0.99 0.1 ]
    Reward: 0.786
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 14/50
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 0.780
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 15/50
    Selected action: 2
    Config: [0.01 0.99 0.2 ]
    Reward: 0.787
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 16/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.777
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 17/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.762
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 18/50
    Selected action: 28
    Config: [0.3  0.99 0.05]
    Reward: 0.777
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 19/50
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 0.789
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 20/50
    Selected action: 32
    Config: [0.3  0.99 0.4 ]
    Reward: 0.786
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 21/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.775
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.787
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Selected action: 6
    Config: [0.01 0.99 0.6 ]
    Reward: 0.797
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 24/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.773
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 25/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.777
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 26/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.772
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 27/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.783
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 28/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.781
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 29/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 30/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.784
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Selected action: 8
    Config: [0.02 0.99 0.1 ]
    Reward: 0.772
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 32/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.768
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 33/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.776
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 34/50
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 0.785
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 35/50
    Selected action: 39
    Config: [0.5  0.99 0.4 ]
    Reward: 0.777
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
  Trial 36/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.785
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 37/50
    Selected action: 32
    Config: [0.3  0.99 0.4 ]
    Reward: 0.770
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.767
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 39/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.784
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 40/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.769
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 41/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.784
    Best config so far: [0.1  0.99 0.4 ]
    Best performance so far: 0.797
    Meta-environment terminated early, but continuing HPO...
  Trial 42/50
    Selected action: 38
    Config: [0.5  0.99 0.3 ]
    Reward: 0.800
    *** NEW BEST! ***
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
  Trial 43/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.771
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
    Meta-environment terminated early, but continuing HPO...
  Trial 44/50
    Selected action: 6
    Config: [0.01 0.99 0.6 ]
    Reward: 0.789
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.777
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
    Meta-environment terminated early, but continuing HPO...
  Trial 46/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.780
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
    Meta-environment terminated early, but continuing HPO...
  Trial 47/50
    Selected action: 2
    Config: [0.01 0.99 0.2 ]
    Reward: 0.790
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
    Meta-environment terminated early, but continuing HPO...
  Trial 48/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.780
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
    Meta-environment terminated early, but continuing HPO...
  Trial 50/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.776
    Best config so far: [0.5  0.99 0.3 ]
    Best performance so far: 0.8
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.5  0.99 0.3 ]
Best performance: 0.800
Total trials completed: 50

Run 5
Trial 1/50
    Selected action: 34
    Config: [0.3  0.99 0.6 ]
    Reward: 0.773
    *** NEW BEST! ***
    Best config so far: [0.3  0.99 0.6 ]
    Best performance so far: 0.773
  Trial 2/50
    Selected action: 23
    Config: [0.1  0.99 0.2 ]
    Reward: 0.772
    Best config so far: [0.3  0.99 0.6 ]
    Best performance so far: 0.773
  Trial 3/50
    Selected action: 40
    Config: [0.5  0.99 0.5 ]
    Reward: 0.786
    *** NEW BEST! ***
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
  Trial 4/50
    Selected action: 39
    Config: [0.5  0.99 0.4 ]
    Reward: 0.780
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
  Trial 5/50
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 0.782
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
  Trial 6/50
    Selected action: 22
    Config: [0.1  0.99 0.1 ]
    Reward: 0.772
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
  Trial 7/50
    Selected action: 21
    Config: [0.1  0.99 0.05]
    Reward: 0.771
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
  Trial 8/50
    Selected action: 19
    Config: [0.05 0.99 0.5 ]
    Reward: 0.774
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
  Trial 9/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.775
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
  Trial 10/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.781
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
  Trial 11/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.784
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 12/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.773
    Best config so far: [0.5  0.99 0.5 ]
    Best performance so far: 0.7859999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 13/50
    Selected action: 16
    Config: [0.05 0.99 0.2 ]
    Reward: 0.791
    *** NEW BEST! ***
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
  Trial 14/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.786
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 15/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.767
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 16/50
    Selected action: 11
    Config: [0.02 0.99 0.4 ]
    Reward: 0.787
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
  Trial 17/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 18/50
    Selected action: 20
    Config: [0.05 0.99 0.6 ]
    Reward: 0.784
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 19/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.774
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 20/50
    Selected action: 39
    Config: [0.5  0.99 0.4 ]
    Reward: 0.790
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 21/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.779
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 22/50
    Selected action: 39
    Config: [0.5  0.99 0.4 ]
    Reward: 0.781
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 23/50
    Selected action: 7
    Config: [0.02 0.99 0.05]
    Reward: 0.776
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
  Trial 24/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.771
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 25/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.769
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 26/50
    Selected action: 29
    Config: [0.3  0.99 0.1 ]
    Reward: 0.776
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
  Trial 27/50
    Selected action: 8
    Config: [0.02 0.99 0.1 ]
    Reward: 0.778
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
  Trial 28/50
    Selected action: 35
    Config: [0.5  0.99 0.05]
    Reward: 0.786
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
  Trial 29/50
    Selected action: 4
    Config: [0.01 0.99 0.4 ]
    Reward: 0.790
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
  Trial 30/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 31/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.785
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 32/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.782
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 33/50
    Selected action: 32
    Config: [0.3  0.99 0.4 ]
    Reward: 0.783
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
  Trial 34/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.768
    Best config so far: [0.05 0.99 0.2 ]
    Best performance so far: 0.791
    Meta-environment terminated early, but continuing HPO...
  Trial 35/50
    Selected action: 13
    Config: [0.02 0.99 0.6 ]
    Reward: 0.793
    *** NEW BEST! ***
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
  Trial 36/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.783
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 37/50
    Selected action: 24
    Config: [0.1  0.99 0.3 ]
    Reward: 0.769
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 38/50
    Selected action: 17
    Config: [0.05 0.99 0.3 ]
    Reward: 0.782
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
  Trial 39/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.791
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 40/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.779
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 41/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.776
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 42/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.785
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 43/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.775
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 44/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.770
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 45/50
    Selected action: 37
    Config: [0.5  0.99 0.2 ]
    Reward: 0.785
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
  Trial 46/50
    Selected action: 16
    Config: [0.05 0.99 0.2 ]
    Reward: 0.773
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 47/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.783
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 48/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.777
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 49/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.792
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    Meta-environment terminated early, but continuing HPO...
  Trial 50/50
    Selected action: 9
    Config: [0.02 0.99 0.2 ]
    Reward: 0.779
    Best config so far: [0.02 0.99 0.6 ]
    Best performance so far: 0.7929999999999999
    HPO completed after 50 trials

=== FINAL RESULTS ===
Best configuration: [0.02 0.99 0.6 ]
Best performance: 0.793
Total trials completed: 50
"""

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