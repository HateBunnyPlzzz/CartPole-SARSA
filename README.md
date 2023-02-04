SARSA 

It's a model free meaning you do not need a complete model of your environment to actually get some learning done and that's improtant because there's many cases in which you dont know the full model of the environment that means you dont know the state-transition probabilities so if you're in some state S and take action A what is the prob we'll end up in state S-prime and get reward R, those prob are not completely known for all problems and so algorithms that handle that uncertainty are critical for real-world applications.

Another neat things is that this is a bootstrap method meaning it uses estimates to generate other estimates.

