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
#
# b = Book('The Python Book')
# print(b.title)


class PositiveInterger(int):
    def __init__(self,value):
        super(PositiveInterger, self).__init__(self,abs(value))

i = PositiveInterger(-3)
print(i)

class PositiveInteger(int):
    def __new__(cls, value):
        return super(PositiveInteger, cls).__new__(cls, abs(value))

i = PositiveInteger(-3)
print(i)

class Person(object):
    def __new__(cls,name,age):
        print('This is __new__')
        return super(Person,cls).__new__(cls )

    def __init__(self,name,age):
        print('This is __init__')
        self.name = name
        self.age = age

    def __str__(self):
        return '<Person: %s(%s)>' %(self.name,self.age)


if __name__ == '__main__':
    zhangsan = Person('zhangsan',24)
    print(zhangsan)
