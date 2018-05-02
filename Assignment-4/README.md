#Instruction for homework assignment 4

**Make sure to download GoogleNews-vectors-negative300.bin <-- this is an open source file can be found anywhere on webstie**

  python3 features.py:

    This command will produce the part 1 and also part 2. Below the features.py,
    there are two commands: features_stub, and features_stub_2. features_stub is for
    part 1 in assignment 4. features_stub_2 is for part 2 in the assignment 4.

    Note that: I comment out the classifier so that it doesn't overlap each other.
    If you want to run only for DT model, you need to comment out the rest of the classifier
    and uncomment the DT model classifier.

  python3 blogs-competition-P2.py

    This command will evaluate the pickle file for competition. To generate the competition
    pickle file, you must run features.py first. (specifically features_stub_2). The result
    will be printed on final-result.txt

  all-results.txt:
  
    this file contains most of the results that I generated through the experiment
    from part 1 and part 2. I believe that it's hard to count every experiment you do and store into the
    .txt file.

  competition_pickle:
  
    for competition, I am using the naive bayes from second part of the assignment 4. The feature that
    I selected were word_bin and word_features. Due to lack of time, I never had a chance to do the
    feature seletion, but I believe that will get me result of 70% or above.

  
 
