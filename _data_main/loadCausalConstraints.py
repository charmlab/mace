# TODO: ideally, you would write a new class called CausalModel, M = {X, U, F}
#       where perhaps F can be defined parametrically (e.g., in the case of linear
#       models). This class would have a method that automatically generates the
#       required causal consistency constraints for pysmt.


from pysmt.shortcuts import *
from pysmt.typing import *

def getGermanCausalConsistencyConstraints(model_symbols, factual_sample):
  # Gender (no parents)
  g = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x0']['symbol'],
        factual_sample['x0'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x0']['symbol'],
      model_symbols['interventional']['x0']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x0']['symbol'],
      factual_sample['x0'],
    ),
  )

  # Age (no parents)
  a = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x1']['symbol'],
        factual_sample['x1'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x1']['symbol'],
      model_symbols['interventional']['x1']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x1']['symbol'],
      factual_sample['x1'],
    ),
  )

  # Credit (parents: age, sex)
  c = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        ToReal(model_symbols['interventional']['x2']['symbol']),
        ToReal(factual_sample['x2']),
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      ToReal(model_symbols['counterfactual']['x2']['symbol']),
      ToReal(model_symbols['interventional']['x2']['symbol']),
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      ToReal(model_symbols['counterfactual']['x2']['symbol']),
      Plus([
        ToReal(factual_sample['x2']),
        Times(
          Minus(
            ToReal(model_symbols['counterfactual']['x0']['symbol']),
            ToReal(factual_sample['x0']),
          ),
          # If you want to support numeric-int children, then you should round
          # these structural equation weights.
          # Real(float(552.43925387))
          Real(float(550))
        ),
        Times(
          Minus(
            ToReal(model_symbols['counterfactual']['x1']['symbol']),
            ToReal(factual_sample['x1']),
          ),
          # If you want to support numeric-int children, then you should round
          # these structural equation weights.
          # Real(float(4.4847736))
          Real(float(4.5))
        ),
      ])
    ),
  )

  # Repayment duration (parents: credit)
  r = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        ToReal(model_symbols['interventional']['x3']['symbol']),
        ToReal(factual_sample['x3']),
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      ToReal(model_symbols['counterfactual']['x3']['symbol']),
      ToReal(model_symbols['interventional']['x3']['symbol']),
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      ToReal(model_symbols['counterfactual']['x3']['symbol']),
      Plus([
        ToReal(factual_sample['x3']),
        Times(
          Minus(
            ToReal(model_symbols['counterfactual']['x2']['symbol']),
            ToReal(factual_sample['x2']),
          ),
          # If you want to support numeric-int children, then you should round
          # these structural equation weights.
          # Real(float(0.00266995))
          Real(float(0.0025))
        ),
      ])
    ),
  )

  return And([g,a,c,r])





















# import numpy as np
# import pandas as pd

# import loadData

# from random import seed
# RANDOM_SEED = 54321
# seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
# np.random.seed(RANDOM_SEED)

# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso

# dataset_obj = loadData.loadDataset('random', return_one_hot = False, load_from_cache = False)
# df = dataset_obj.data_frame_kurz

# # See Figure 3 in paper

# # node: x1     parents: {x0}
# X_train = df[['x0']]
# y_train = df[['x1']]
# model_pretrain = LinearRegression()
# # model_pretrain = Lasso()
# model_trained = model_pretrain.fit(X_train, y_train)
# print(model_trained.coef_)
# print(model_trained.intercept_)

# # node: x2     parents: {x1,  x2}
# X_train = df[['x0', 'x1']]
# y_train = df[['x2']]
# model_pretrain = LinearRegression()
# model_trained = model_pretrain.fit(X_train, y_train)
# print(model_trained.coef_)
# print(model_trained.intercept_)
# #


# def getRandomCausalConsistencyConstraints(model_symbols, factual_sample):
#   a = Ite(
#     Not( # if YES intervened
#       EqualsOrIff(
#         model_symbols['interventional']['x0']['symbol'],
#         factual_sample['x0'],
#       )
#     ),
#     EqualsOrIff( # set value of X^CF to the intervened value
#       model_symbols['counterfactual']['x0']['symbol'],
#       model_symbols['interventional']['x0']['symbol'],
#     ),
#     EqualsOrIff( # else, set value of X^CF to (8) from paper
#       model_symbols['counterfactual']['x0']['symbol'],
#       factual_sample['x0'],
#     ),
#   )

#   b = Ite(
#     Not( # if YES intervened
#       EqualsOrIff(
#         model_symbols['interventional']['x1']['symbol'],
#         factual_sample['x1'],
#       )
#     ),
#     EqualsOrIff( # set value of X^CF to the intervened value
#       model_symbols['counterfactual']['x1']['symbol'],
#       model_symbols['interventional']['x1']['symbol'],
#     ),
#     EqualsOrIff( # else, set value of X^CF to (8) from paper
#       model_symbols['counterfactual']['x1']['symbol'],
#       Plus(
#         factual_sample['x1'],
#         Minus(
#           model_symbols['counterfactual']['x0']['symbol'],
#           factual_sample['x0'],
#         )
#       )
#     ),
#   )

#   c = Ite(
#     Not( # if YES intervened
#       EqualsOrIff(
#         model_symbols['interventional']['x2']['symbol'],
#         factual_sample['x2'],
#       )
#     ),
#     EqualsOrIff( # set value of X^CF to the intervened value
#       model_symbols['counterfactual']['x2']['symbol'],
#       model_symbols['interventional']['x2']['symbol'],
#     ),
#     EqualsOrIff( # else, set value of X^CF to (8) from paper
#       model_symbols['counterfactual']['x2']['symbol'],
#       Plus(
#         factual_sample['x2'],
#         Minus(
#           Plus(
#             Times(
#               Minus(
#                 model_symbols['counterfactual']['x0']['symbol'],
#                 Real(1)
#               ),
#               Real(0.25)
#             ),
#             Times(
#               model_symbols['counterfactual']['x1']['symbol'],
#               Real(float(np.sqrt(3)))
#             )
#           ),
#           Plus(
#             Times(
#               Minus(
#                 factual_sample['x0'],
#                 Real(1)
#               ),
#               Real(0.25)
#             ),
#             Times(
#               factual_sample['x1'],
#               Real(float(np.sqrt(3)))
#             )
#           ),
#         )
#       )
#     ),
#   )

#   return And([a,b,c])

# TODO: this function needs to be updated to add the weights for linear scm_model;
#       see process_german_data.py for an example. Also, perhaps it is already
#       implemented (but commented) above for some reason...
def getRandomCausalConsistencyConstraints(model_symbols, factual_sample):
  a = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x0']['symbol'],
        factual_sample['x0'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x0']['symbol'],
      model_symbols['interventional']['x0']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x0']['symbol'],
      factual_sample['x0'],
    ),
  )

  b = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x1']['symbol'],
        factual_sample['x1'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x1']['symbol'],
      model_symbols['interventional']['x1']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x1']['symbol'],
      factual_sample['x1'],
    ),
  )

  c = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x2']['symbol'],
        factual_sample['x2'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x2']['symbol'],
      model_symbols['interventional']['x2']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x2']['symbol'],
      factual_sample['x2'],
    ),
  )

  return And([a,b,c])



























# def getMortgageCausalConsistencyConstraints(model_symbols, factual_sample):
#   a = Ite(
#     Not( # if YES intervened
#       EqualsOrIff(
#         model_symbols['interventional']['x0']['symbol'],
#         factual_sample['x0'],
#       )
#     ),
#     EqualsOrIff( # set value of X^CF to the intervened value
#       model_symbols['counterfactual']['x0']['symbol'],
#       model_symbols['interventional']['x0']['symbol'],
#     ),
#     EqualsOrIff( # else, set value of X^CF to (8) from paper
#       model_symbols['counterfactual']['x0']['symbol'],
#       factual_sample['x0'],
#     ),
#   )

#   b = Ite(
#     Not( # if YES intervened
#       EqualsOrIff(
#         model_symbols['interventional']['x1']['symbol'],
#         factual_sample['x1'],
#       )
#     ),
#     EqualsOrIff( # set value of X^CF to the intervened value
#       model_symbols['counterfactual']['x1']['symbol'],
#       model_symbols['interventional']['x1']['symbol'],
#     ),
#     EqualsOrIff( # else, set value of X^CF to (8) from paper
#       model_symbols['counterfactual']['x1']['symbol'],
#       Plus(
#         factual_sample['x1'],
#         Times(
#           Minus(
#             ToReal(model_symbols['counterfactual']['x0']['symbol']),
#             ToReal(factual_sample['x0']),
#           ),
#           Real(0.3)
#         ),
#       )
#     ),
#   )

#   return And([a,b])

def getMortgageCausalConsistencyConstraints(model_symbols, factual_sample):
  a = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x0']['symbol'],
        factual_sample['x0'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x0']['symbol'],
      model_symbols['interventional']['x0']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x0']['symbol'],
      factual_sample['x0'],
    ),
  )

  b = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x1']['symbol'],
        factual_sample['x1'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x1']['symbol'],
      model_symbols['interventional']['x1']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x1']['symbol'],
      factual_sample['x1'],
    ),
  )

  return And([a,b])
