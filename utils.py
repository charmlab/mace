import inspect

# See https://www.python-course.eu/python3_memoization.php
class Memoize:

  def __init__(self, fn):

    self.fn = fn
    self.memo = {}

  def __call__(self, *args, **kwargs):

    # to support default args: https://docs.python.org/3.3/library/inspect.html
    sig = inspect.signature(self.fn)
    ba = sig.bind(*args)
    for param in sig.parameters.values():
      if param.name not in ba.arguments:
        ba.arguments[param.name] = param.default
    args = ba.args

    if args not in self.memo:
      self.memo[args] = self.fn(*args)
    return self.memo[args]
