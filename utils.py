import inspect
import collections

# See https://www.python-course.eu/python3_memoization.php
class Memoize:

  def __init__(self, fn):

    self.fn = fn
    self.memo = {}

  def __call__(self, *args, **kwargs):


    sig = inspect.signature(self.fn)
    ba = sig.bind(*args)
    for param in sig.parameters.values():
      # to support default args: https://docs.python.org/3.3/library/inspect.html
      if param.name not in ba.arguments:
        ba.arguments[param.name] = param.default
      # For some reason, when we pass in something like tmp(a,b=1) to the function
      # def tmp(a,b=0), the args will remain (a,0) but not (a,1). Therefore, we
      # should update `ba` according to the k,v pairs in kwargs
      if param.name in kwargs.keys():
        ba.arguments[param.name] = kwargs[param.name]
    args = ba.args

    # convert lists and numpy array into tuples so that they can be used as keys
    hashable_args = tuple([
      arg if isinstance(arg, collections.Hashable) else str(arg)
      for arg in args
    ])

    if hashable_args not in self.memo:
      self.memo[hashable_args] = self.fn(*args)
    return self.memo[hashable_args]
