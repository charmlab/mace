

# See https://www.python-course.eu/python3_memoization.php
# TODO: add support for keyword arguments: https://stackoverflow.com/a/37858058
class Memoize:
  def __init__(self, fn):
    self.fn = fn
    self.memo = {}
  def __call__(self, *args):
    if args not in self.memo:
      self.memo[args] = self.fn(*args)
    return self.memo[args]
