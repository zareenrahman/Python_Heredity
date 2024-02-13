Write an AI to assess the likelihood that a person will have a particular genetic trait.

Introduction

Mutated versions of the GJB2 gene are among the leading causes of hearing impairment in newborns. Each person carries two versions of the gene, so each person has the potential to possess either 0, 1, or 2 copies of the hearing impairment version GJB2.  Unless a person undergoes genetic testing, it’s not so easy to know how many copies of mutated GJB2 a person has. This is some “hidden state”: information that has an effect that we can observe (hearing impairment), but that we don’t necessarily directly know. After all, some people might have 1 or 2 copies of mutated GJB2 but not exhibit hearing impairment, while others might have no copies of mutated GJB2 yet still exhibit hearing impairment.

Every child inherits one copy of the GJB2 gene from each of their parents. If a parent has two copies of the mutated gene, then they will pass the mutated gene on to the child; if a parent has no copies of the mutated gene, then they will not pass the mutated gene on to the child; and if a parent has one copy of the mutated gene, then the gene is passed on to the child with probability 0.5. After a gene is passed on, though, it has some probability of undergoing additional mutation: changing from a version of the gene that causes hearing impairment to a version that doesn’t, or vice versa.

We can attempt to model all of these relationships by forming a Bayesian Network of all the relevant variables. In this network, each person has a Gene random variable representing how many copies of a particular gene a person has which can be either 0, 1 or 2 copies of the gene. Each person in the family also has a Trait random variable, which can be yes or no depending on whether that person expresses a trait (hearing impairment) caused by the gene.

![image](https://github.com/zareenrahman/Python_Heredity/assets/155265507/7b5ac2c4-d7cc-42e5-8365-f429647f27b9)

Bayesian Network Model

In the model we are using, each person's Gene random variable has a conditional probability based on their parent's Gene variable. If the person is a root of the family tree (no parent information), then the Gene random variable is given an unconditional probability for having 0, 1 or 2 Genes.

Inference by Enumeration
Given a family tree with several members, some being parents to others, and some knowledge of certain family members having the genetic trait or not, how can the conditional probabilities of each family member having 0, 1 or 2 copies of the gene be calculated?

One way to do this is using Inference by enumeration, which can be expressed using the equation below:

Equation:

![image](https://github.com/zareenrahman/Python_Heredity/assets/155265507/73de2ae2-f78f-4200-b879-3abeda5c5952)

where:

-	X is the query variable
-	e is the evidence (knowledge we hold about the family)
-	P(X | e) is the probability distribution of variable X given knowledge e
-	α is a normalisation factor (the sum of probabilities for query variable X must be 1)
-	y ranges over all values of all hidden variables

The query variable is the variable for which we are trying to compute/predict the probability. In this problem, this query variable could be the probability that the child exhibits the trait. The evidence then could be the observations we make about the child’s parents on whether they exhibit the trait. Finally, the hidden variables could in this case be the number of copies of genes the parents or the child have. To draw an analogy to help understand the normalization constant α, recall the law you already know: P(A,B)=P(A|B)*P(B). In that case, the joint probability was scaled/normalized by P(B)

In essence, to find the probability distribution for the number of genes each person possesses, and whether or not they will have the trait (if not already known), we must:

1. Generate every possible combination of genes and traits for the family.
2. Discard any combination that disagrees with the known evidence about the family.
3. Calculate the joint probability of each combination.
4. Add the joint probability of each combination to each person's probability of having 0,1 or 2 Genes and having the trait or not.
5. Normalize each person's Gene probability distribution and trait probability distribution.

Understanding the Environment 

Look at one of the sample data sets in the data directory by opening data/family0.csv (you can open it up in a text editor or a spreadsheet application like Google Sheets, Excel, or Apple Numbers). Notice that the first row defines the columns for this CSV file: name, mother, father, and trait. The next row indicates that Harry has Lily as a mother, and James as a father, and the empty cell for trait means we don’t know whether Harry has the trait or not. James, meanwhile, has no parents listed in our data set (as indicated by the empty cells for mother and father), and does exhibit the trait (as indicated by the 1 in the trait cell). Lily, on the other hand, also has no parents listed in the data set but does not exhibit the trait (as indicated by the 0 in the trait cell). 

Open heredity.py and look first at the definition of PROBS. PROBS is a dictionary containing several constants representing probabilities of various events. All these events have to do with how many copies of a particular gene a person has (hereafter referred to as simply “the gene”), and whether a person exhibits a particular trait (hereafter referred to as “the trait”) based on that gene. The data here is loosely based on the probabilities for the hearing impairment version of the GJB2 gene and the hearing impairment trait, but by changing these values, you could use your AI to draw inferences about other genes and traits as well! 

First, PROBS["gene"] represents the unconditional probability distribution over the gene (i.e., the probability if we know nothing about that person’s parents). Based on the data in the distribution code, in the population, there’s a 1% chance of having 2 copies of the gene, a 3% chance of having 1 copy of the gene, and a 96% chance of having 0 copies of the gene. 

Next, PROBS["trait"] represents the conditional probability that a person exhibits a trait (like hearing impairment). This is three different probability distributions: one for each possible value for the gene. So PROBS["trait"][2] is the probability distribution that a person has the trait given that they have two versions of the gene: in this case, they have a 65% chance of exhibiting the trait and a 35% chance of not exhibiting the trait. Meanwhile, if a person has 0 copies of the gene, they have a 1% chance of exhibiting the trait and a 99% chance of not exhibiting the trait. 

Finally, PROBS["mutation"] is the probability that a gene mutates from being the gene in question to not being that gene, and vice versa. If a mother has two versions of the gene, for example, and therefore passes one on to her child, there’s a 1% chance it mutates into not being the target gene anymore. Conversely, if a mother has no versions of the gene, and therefore does not pass it on to her child, there’s a 1% chance it mutates into being the target gene. It’s therefore possible that even if neither parent has any copies of the gene in question, their child might have 1 or even 2 copies of the gene. Ultimately, the probabilities you calculate will be based on these values in PROBS. 

Now, look at the main function. The function first loads data from a file into a dictionary people. people map each person’s name to another dictionary containing information about them: including their name, their mother (if one is listed in the data set), their father (if one is listed in the data set), and whether they are observed to have the trait in question (True if they do, False if they don’t, and None if we don’t know). 

Next, main defines a dictionary of probabilities, with all probabilities initially set to 0. This is ultimately what your project will compute: for each person, your AI will calculate the probability distribution over how many of copies of the gene they have, as well as whether they have the trait or not. probabilities["Harry"]["gene"][1], for example, will be the probability that Harry has 1 copy of the gene, and probabilities["Lily"]["trait "][False] will be the probability that Lily does not exhibit the trait. If unfamiliar, this probabilities dictionary is created using a Python dictionary comprehension, which in this case creates one key/value pair for each person in our dictionary of people. 

Ultimately, we’re looking to calculate these probabilities based on some evidence: given that we know certain people do or do not exhibit the trait, we’d like to determine these probabilities. Recall that we can calculate a conditional probability by summing up all the joint probabilities that satisfy the evidence, and then normalize those probabilities so that they each sum to 1. Your task in this project is to implement three functions to do just that: joint_probability to compute a joint probability, update to add the newly computed joint probability to the existing probability distribution and then normalize to ensure all probability distributions sum to 1 at the end.

Specification
Complete the implementations of joint_probability, update, and normalize.
The joint_probability function should take as input a dictionary of people, along with data about who has how many copies of each of the genes, and who exhibits the trait. The function should return the joint probability of all of those events taking place.

The function accepts four values as input: people, one_gene, two_genes, and have_trait.
•	people is a dictionary of people. The keys represent names, and the values are dictionaries that contain mother and father keys. You may assume that either mother and father are both blank (no parental information in the data set), or mother and father will both refer to other people in the people dictionary.
o	one_gene is a set of all people for whom we want to compute the probability that they have one copy of the gene.
o	two_genes is a set of all people for whom we want to compute the probability that they have two copies of the gene.
o	have_trait is a set of all people for whom we want to compute the probability that they have the trait.
o	For any person not in one_gene or two_genes, we would like to calculate the probability that they have no copies of the gene; and for anyone not in have_trait, we would like to calculate the probability that they do not have the trait.
•	For example, if the family consists of Harry, James, and Lily, then calling this function where one_gene = {"Harry"}, two_genes = {"James"}, and trait = {"Harry", "James"} should calculate the probability that Lily has zero copies of the gene, Harry has one copy of the gene, James has two copies of the gene, Harry exhibits the trait, James exhibits the trait, and Lily does not exhibit the trait.
•	For anyone with no parents listed in the data set, use the probability distribution PROBS["gene"] to determine the probability that they have a particular number of the gene. For anyone with parents in the data set, each parent will pass one of their two genes on to their child randomly, and there is a PROBS["mutation"] chance that it mutates (goes from being the gene to not being the gene, or vice versa).
•	Use the probability distribution PROBS["trait"] to compute the probability that a person does or does not have a particular trait.

The update function adds a new joint distribution probability to the existing probability distributions in probabilities.

•	The function accepts five values as input: probabilities, one_gene, two_genes, have_trait, and p.
o	probabilities is a dictionary of people. Each person is mapped to a "gene" distribution and a "trait" distribution.
•	one_gene is a set of people with one copy of the gene in the current joint distribution.
•	two_genes is a set of people with two copies of the gene in the current joint distribution.
•	have_trait is a set of people with the trait in the current joint distribution.
•	p is the probability of the joint distribution.
•	For each person person in probabilities, the function should update the probabilities[person]["gene"] distribution and probabilities[person]["trait"] distribution by adding p to the appropriate value in each distribution. All other values should be left unchanged.
o	For example, if "Harry" were in both two_genes and in have_trait, then p would be added to probabilities["Harry"]["gene"][2] and to probabilities["Harry"]["trait"][True].
•	The function should not return any value: it just needs to update the probabilities dictionary. The normalize function updates a dictionary of probabilities such that each probability distribution is normalized (i.e., sums to 1, with relative proportions the same).
•	The function accepts a single value: probabilities.
o	probabilities is a dictionary of people. Each person is mapped to a "gene" distribution and a "trait" distribution.
•	For both distributions for each person in probabilities, this function should normalize that distribution so that the values in the distribution sum to 1, and the relative values in the distribution are the same.
•	For example, if probabilities["Harry”] ["trait"][True] were equal to 0.1 and probabilities["Harry"]["trait"][False] were equal to 0.3, then your function should update the former value to be 0.25 and the latter value to be 0.75: the numbers now sum to 1, and the latter value is still three times larger than the former value.
•	The function should not return any value: it just needs to update the probabilities dictionary.

Implementation Hints

Recall that to compute a joint probability of multiple events, you can do so by multiplying those probabilities together. But remember that for any child, the probability of them having a certain number of genes is conditional on what genes their parents have.

Submission Guidelines

Each dataset has people that don’t have any value for the trait. You should not need to modify anything  else in heredity.py other than the three functions the specification calls for you to implement, though you may write additional functions and/or import other Python standard library modules. You may also import numpy, if you are familiar with it, but you should not need to use any other third-party Python modules.

Usage

Requires Python(3) to run:

$python heredity.py (data/familyX.csv)
