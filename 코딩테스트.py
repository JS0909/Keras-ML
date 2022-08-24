

def solution(number, k):
    
    num_str = str(number)
    nl = list(num_str)
    
    for i in range(len(nl)):
        nl[i] = int(nl[i])
        
    count = len(nl)-k
    
    
    ans_num=[]
    for i in range(len(nl)-k):
        ans_num.append(str(nl[i]))
    
    
    answer = ''.join(ans_num)
    return answer
    
print(solution(1924,2))
print(solution(1231234,3))
print(solution(4177252841,4))