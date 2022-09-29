def a():
    while 1:        
        yield 0
        yield 1
           
for x in a():
    print(x)