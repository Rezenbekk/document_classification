from enum import Enum


class Label(Enum):
    none = 0
    science = 1
    style = 2
    culture = 3
    life = 4
    economics = 5
    business = 6
    travel = 7
    forces = 8
    media = 9
    sport = 10


class Document:
    body = ''
    category = Label.none

    def __init__(self, _category, _title, _body):
        if _category == '':
            self.category = Label.none
        else:
            self.category = Label[_category]
        self.body = _title + ' ' + _body

print("Custom file Document.py has been imported.")
