class Fruit:
    def name(self):
        print("fruit_name")

class Apple(Fruit):
    def name(self):
        print('apple')
        
F = Fruit()
A = Apple()

F.name()
A.name()
