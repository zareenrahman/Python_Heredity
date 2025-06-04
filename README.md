# Heredity: Bayesian Genetic Trait Prediction

This project uses **Bayesian inference** to estimate the likelihood that family members carry a mutated gene and express a related trait. It works with family trees and simulates inheritance patterns across generations.

---
## Introduction

Mutations in the GJB2 gene are a major cause of hearing impairment. Each person has 0–2 copies of the gene, which they inherit from their parents. The presence of the gene doesn't always guarantee the trait, making it a “hidden state.”

This tool models inheritance using a **Bayesian Network**, where each person has a `Gene` variable (0, 1, or 2) and a `Trait` variable (True/False). A small mutation probability adds realism. In detail, we can attempt to model these relationships by forming a Bayesian Network of all the relevant variables. In this network, each person has a Gene random variable representing how many copies of a particular gene a person has, which can be either 0, 1, or 2 copies of the gene. Each person in the family also has a Trait random variable, which can be yes or no, depending on whether that person expresses a trait (hearing impairment) caused by the gene.

![image](https://github.com/zareenrahman/Python_Heredity/assets/155265507/7b5ac2c4-d7cc-42e5-8365-f429647f27b9)

---

## Bayesian Network Model

In our model, each person's Gene random variable has a conditional probability based on their parents' Gene variable. If the person is a root of the family tree (no parent information), then the Gene random variable is given an unconditional probability for having 0, 1, or 2 Genes.

## Inference by Enumeration

To predict each person's gene/trait probability:
1. Enumerate all gene/trait combinations.
2. Discard those contradicting known evidence.
3. Calculate the joint probability of each valid case.
4. Add the results to the cumulative totals.
5. Normalize for valid probability distributions.

## Equation:

![image](https://github.com/zareenrahman/Python_Heredity/assets/155265507/73de2ae2-f78f-4200-b879-3abeda5c5952)

where:

-	X is the query variable
-	e is the evidence (knowledge we hold about the family)
-	P(X | e) is the probability distribution of variable X given knowledge e
-	α is a normalisation factor (the sum of probabilities for query variable X must be 1)
-	y ranges over all values of all hidden variables

## Understanding the Environment

Each dataset (e.g., `data/family0.csv`) lists people with parents and trait status. Missing values mean unknown data.
Example:
* `Harry` has parents `Lily` and `James` but no trait info.
* `James` and `Lily` have no parents and known traits (0 or 1).
The program estimates probabilities of:
* having 0, 1, or 2 mutated genes
* expressing the trait

We use these constants:
```python
PROBS = {
  "gene": {2: 0.01, 1: 0.03, 0: 0.96},
  "trait": {
    2: {True: 0.65, False: 0.35},
    1: {True: 0.56, False: 0.44},
    0: {True: 0.01, False: 0.99}
  },
  "mutation": 0.01
}
```
If a person has no parents listed, their gene distribution comes from `PROBS["gene"]`.
If they have parents, their gene count depends on inherited probabilities and mutation.

### What Happens in `heredity.py`:

* Loads the CSV dataset
* Sets up a nested dictionary of probabilities
* Calculates joint probabilities of all possible valid configurations
* Updates and normalizes the result

### Core Functions:

* `joint_probability`: Computes the full probability for a gene-trait configuration.
* `update`: Adds probability to each person’s gene/trait totals.
* `normalize`: Ensures final probabilities sum to 1.
---

## Specification

We have implemented here:

* `joint_probability`
* `update`
* `normalize`

These functions handle computing, accumulating, and normalizing probabilities based on genetic rules and trait observations.

---

## Usage

Ensure you have Python 3 installed.
Run the program using any of the family datasets:

```bash
python heredity.py data/family0.csv
python heredity.py data/family1.csv
python heredity.py data/family2.csv
python heredity.py data/family3.csv
python heredity.py data/family4.csv
```
Each file contains a different family tree structure with varying amounts of known and unknown gene/trait data.
The output will show for each person:

* The probability of having 0, 1, or 2 copies of the gene.
* The probability of expressing or not expressing the trait.
