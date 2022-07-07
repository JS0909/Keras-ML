class Fruit:
    def __init__(self, name):
        self.name = name
              
class This_year(Fruit):
    def __init__(self, name):
        super().__init__(name)
        self.grow = "2022"
        print('연도:', self.grow, ', 이름:', self.name)
        
a = This_year('apple')
print(a)