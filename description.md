# Overview

In this challenge, your task will be to predict the social role of users on a social network. We define two key social roles: "Influencers" and "Observers". Influencers are characterized by a high-leverage, asymmetrical network (a high follower-to-following ratio), while Observers maintain a more balanced, reciprocal network.

From the original dataset, all explicit network metrics (like follower and following counts) have been removed. Each user is associated with a collection of tweets. Given this set of users and their text, your job is to predict which ones belong to the "Influencer" class.

# Description

This competition explores the connection between language and social network structure. On any social platform, users naturally adopt different roles. Some act as "Influencers" or "broadcasters", accumulating a large audience (defined by a high follower-to-following ratio) to amplify their message. Others act as "Observers", engaging in reciprocal conversations to build a community (defined by a more balanced ratio).

This challenge asks: is a user's network strategy encoded in their posts? Your goal is to build a model that can determine a user's social role (Influencer or Observer) based only on the tweets they post.

Note: In this dataset, the target label is provided per tweet, not per user. Each tweet is individually labeled as originating from an Influencer (1) or an Observer (0), even though users may post multiple tweets. The aim is to build a model that can predict a user’s social role based on a single tweet.

# The Data

You will be provided with a dataset where each row corresponds to a single tweet.

Training set: 38,560 users, 154,914 tweets, 194 features
Test set: 25,890 users, 103,380 tweets
Target: Binary label (1 = Influencer, 0 = Observer)
Below is a sample of the training data:

# Evaluation

## Evaluation Metric

The performance of your models will be assessed using Accuracy, defined as the proportion of correct predictions:



A correct prediction is one where the predicted label (0 = Observer, 1 = Influencer) matches the true label.

## Submission Format

Submission files should be in .csv format and contain two columns: "ID" and "Predicted".

The "ID" column should contain row indexes (integers starting from zero).
The "Predicted" column should contain the final predicted class (an integer: 0 for "Observer" or 1 for "Influencer") for each user.

## Example Submission File

For each ID in the test set, your file should contain a header and follow this format:

ID,Prediction
0,1
2,1
4,0
8,1
9,0
10,1
13,0
14,0
15,0
etc.