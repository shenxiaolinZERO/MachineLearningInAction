# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'

#The difference between __init__ and __new__

class Book(object):
    def __new__(cls,title):
        print('This is __new__')
        return super(Book,cls).__new__(cls)

    def __init__(self,title):
        print('This is __init__')
        super(Book,self).__init__()
        self.title = title

b = Book('The Python Book')
print(b.title)