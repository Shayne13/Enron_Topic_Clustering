import sys

class EmailUnit(object):

  def __init__(self, owner, sender, recipient, subject, text, processed):
    self.owner = owner
    self.sender = sender
    self.recipient = recipient
    self.subject = subject
    self.text = text
    self.processed = processed

  def __str__(self):
    return 'Owner: {0}, To: {1}, From: {2}'.format(self.owner, self.recipient, self.sender)

  def __repr__(self):
    return str(self)


class SentenceUnit(object):

  def __init__(self, text, processed=None, basic=None):
    self.text = text
    self.processed = processed
    # self.label = label
    # self.index = index
    self.basic = basic

  def __str__(self):
    return u'Original Text: ' + self.text.encode('utf-8')

  def __repr__(self):
    return str(self)

class WordUnit(object):

  def __init__(self, text, token=None, tag=None):
    self.text = text
    self.token = token
    self.score = -1
    self.tag = tag[:2] if tag else None # just first two letters of tag

  def __str__(self):
    return u'Original Text: ' + self.text.encode('utf-8')

  def __repr__(self):
    return str(self)