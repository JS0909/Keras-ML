import numpy as np

def solution(number, k):
    
    nl = list(number)
    # 문자열의 각 요소를 따로 떼어내서 리스트로 만듦
    
    for i in range(len(nl)):
        nl[i] = int(nl[i])
    # 각 요소들을 정수로 바꿈
    
    count = len(nl)-k
    idx = 0
    new_list = []  
    while nl:
        idx += 1  
        now_list = nl[idx:len(nl)-(count-1)]
        now_big=0 
        for i, now in enumerate(now_list):
            if now == 9:
                now_big = now
                iidx = i
                pass
            elif now_big < now:
                now_big = now
                iidx = i       
        new_list.append(now_big)
        idx = idx+iidx                           
        count -= 1                           
        if count==0:                         
            break                            
                                                         
    
    for i in range(len(new_list)):
        new_list[i] = str(new_list[i])

    answer = ''.join(new_list)
    return answer
    
print(solution('1924',2))
print(solution('1231234',3))
print(solution('4177252841',4))