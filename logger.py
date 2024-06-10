class Logger:
  def __init__(self, writter, writter_path):
    self.value = 0
    self.length = 0
    self.writter = writter
    self.writter_path = writter_path
    self.idx = 0
    self.max = None
    self.min = None
    return
    
  def add(self, input, length):
    input /= length
    self.value += input
    self.length += 1
    return

  def write(self, value):
    self.writter.add_scalar(self.writter_path, value, self.idx)

    if self.max == None:
      self.max = value
    if self.min == None:
      self.min = value

    self.max = max(self.max, value)
    self.min = min(self.min, value)
    self.idx += 1
  
  def get(self):
    if self.length == 0:
      return 0
    value = self.value / self.length
    self.value = 0
    self.length = 0

    self.write(value)
    return value
  
  def getMax(self):
    return self.max
  
  def getMin(self):
    return self.min

  def clear(self):
    self.value = 0
    self.length = 0
    self.idx = 0
    self.max = None
    self.min = None
  
  def setPrefix(self, prefix):
    self.write_prefix = prefix

