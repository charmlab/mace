# TODO: ideally, you would write a new class called CausalModel, M = {X, U, F}
#       where perhaps F can be defined parametrically (e.g., in the case of linear
#       models). This class would have a method that automatically generates the
#       required causal consistency constraints for pysmt.


import numpy as np
from pysmt.shortcuts import *
from pysmt.typing import *

def getGermanCausalConsistencyConstraints(model_symbols, factual_sample):
  # Gender (no parents)
  g = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        ToReal(model_symbols['interventional']['x0']['symbol']),
        ToReal(factual_sample['x0']),
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      ToReal(model_symbols['counterfactual']['x0']['symbol']),
      ToReal(model_symbols['interventional']['x0']['symbol']),
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      ToReal(model_symbols['counterfactual']['x0']['symbol']),
      ToReal(factual_sample['x0']),
    ),
  )

  # Age (no parents)
  a = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        ToReal(model_symbols['interventional']['x1']['symbol']),
        ToReal(factual_sample['x1']),
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      ToReal(model_symbols['counterfactual']['x1']['symbol']),
      ToReal(model_symbols['interventional']['x1']['symbol']),
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      ToReal(model_symbols['counterfactual']['x1']['symbol']),
      ToReal(factual_sample['x1']),
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









# TODO: this is the random_linear dataset,... building support for random_nonlinear
# is not possible in SMT... perhaps then build support for the linear approximation
# to the random_nonlinear dataset?
# def getRandomCausalConsistencyConstraints(model_symbols, factual_sample):
#   # x0 (root node; no parents)
#   x0 = Ite(
#     Not( # if YES intervened
#       EqualsOrIff(
#         ToReal(model_symbols['interventional']['x0']['symbol']),
#         ToReal(factual_sample['x0']),
#       )
#     ),
#     EqualsOrIff( # set value of X^CF to the intervened value
#       ToReal(model_symbols['counterfactual']['x0']['symbol']),
#       ToReal(model_symbols['interventional']['x0']['symbol']),
#     ),
#     EqualsOrIff( # else, set value of X^CF to (8) from paper
#       ToReal(model_symbols['counterfactual']['x0']['symbol']),
#       ToReal(factual_sample['x0']),
#     ),
#   )

#   # x1 (parents = {x0})
#   x1 = Ite(
#     Not( # if YES intervened
#       EqualsOrIff(
#         ToReal(model_symbols['interventional']['x1']['symbol']),
#         ToReal(factual_sample['x1']),
#       )
#     ),
#     EqualsOrIff( # set value of X^CF to the intervened value
#       ToReal(model_symbols['counterfactual']['x1']['symbol']),
#       ToReal(model_symbols['interventional']['x1']['symbol']),
#     ),
#     EqualsOrIff( # else, set value of X^CF to (8) from paper
#       ToReal(model_symbols['counterfactual']['x1']['symbol']),
#       Plus(
#         ToReal(factual_sample['x1']),
#         Times(
#           Minus(
#             ToReal(model_symbols['counterfactual']['x0']['symbol']),
#             ToReal(factual_sample['x0']),
#           ),
#           Real(1)
#         )
#       )
#     ),
#   )

#   # x2 (parents = {x0, x1})
#   x2 = Ite(
#     Not( # if YES intervened
#       EqualsOrIff(
#         ToReal(model_symbols['interventional']['x2']['symbol']),
#         ToReal(factual_sample['x2']),
#       )
#     ),
#     EqualsOrIff( # set value of X^CF to the intervened value
#       ToReal(model_symbols['counterfactual']['x2']['symbol']),
#       ToReal(model_symbols['interventional']['x2']['symbol']),
#     ),
#     EqualsOrIff( # else, set value of X^CF to (8) from paper
#       ToReal(model_symbols['counterfactual']['x2']['symbol']),
#       Plus([
#         ToReal(factual_sample['x2']),
#         Times(
#           Minus(
#             ToReal(model_symbols['counterfactual']['x0']['symbol']),
#             ToReal(factual_sample['x0']),
#           ),
#           Real(0.25)
#         ),
#         Times(
#           Minus(
#             ToReal(model_symbols['counterfactual']['x1']['symbol']),
#             ToReal(factual_sample['x1']),
#           ),
#           Real(float(np.sqrt(3)))
#         ),
#       ])
#     ),
#   )

#   return And([x0,x1,x2])


def getRandomCausalConsistencyConstraints(model_symbols, factual_sample):
  # x0 (root node; no parents)
  x0 = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        ToReal(model_symbols['interventional']['x0']['symbol']),
        ToReal(factual_sample['x0']),
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      ToReal(model_symbols['counterfactual']['x0']['symbol']),
      ToReal(model_symbols['interventional']['x0']['symbol']),
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      ToReal(model_symbols['counterfactual']['x0']['symbol']),
      ToReal(factual_sample['x0']),
    ),
  )

  # x1 (parents = {x0})
  x1 = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        ToReal(model_symbols['interventional']['x1']['symbol']),
        ToReal(factual_sample['x1']),
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      ToReal(model_symbols['counterfactual']['x1']['symbol']),
      ToReal(model_symbols['interventional']['x1']['symbol']),
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      ToReal(model_symbols['counterfactual']['x1']['symbol']),
      Plus(
        ToReal(factual_sample['x1']),
        Times(
          Minus(
            ToReal(model_symbols['counterfactual']['x0']['symbol']),
            ToReal(factual_sample['x0']),
          ),
          Real(7.43)
        )
      )
    ),
  )

  # x2 (parents = {x0, x1})
  x2 = Ite(
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
          Real(1.6)
        ),
        Times(
          Minus(
            ToReal(model_symbols['counterfactual']['x1']['symbol']),
            ToReal(factual_sample['x1']),
          ),
          Real(0.015)
        ),
      ])
    ),
  )

  return And([x0,x1,x2])




def getMortgageCausalConsistencyConstraints(model_symbols, factual_sample):
  a = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        ToReal(model_symbols['interventional']['x0']['symbol']),
        ToReal(factual_sample['x0']),
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      ToReal(model_symbols['counterfactual']['x0']['symbol']),
      ToReal(model_symbols['interventional']['x0']['symbol']),
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      ToReal(model_symbols['counterfactual']['x0']['symbol']),
      ToReal(factual_sample['x0']),
    ),
  )

  b = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        ToReal(model_symbols['interventional']['x1']['symbol']),
        ToReal(factual_sample['x1']),
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      ToReal(model_symbols['counterfactual']['x1']['symbol']),
      ToReal(model_symbols['interventional']['x1']['symbol']),
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      ToReal(model_symbols['counterfactual']['x1']['symbol']),
      Plus(
        ToReal(factual_sample['x1']),
        Times(
          Minus(
            ToReal(model_symbols['counterfactual']['x0']['symbol']),
            ToReal(factual_sample['x0']),
          ),
          Real(0.3)
        ),
      )
    ),
  )

  return And([a,b])
