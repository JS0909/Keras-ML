def a():   
    yield 0
    yield 1
           
for x in a():
    print(x)
    
# text = 'text'
# print([text][0])