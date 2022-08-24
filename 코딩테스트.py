import numpy as np

def solution(number, k):
    
    num_str = str(number)
    nl = list(num_str)
    
    for i in range(len(nl)):
        nl[i] = int(nl[i])
    # 숫자를 하나하나 요소의 리스트로 만들었음
    
    count = len(nl)-k
    idx = 0
    new_list = []
    for i in range(len(nl)):
        now_list = nl[idx:len(nl)-(count-1)]
        idx = np.argmax(now_list)+idx
        new_list.append(nl[idx])
        count -= 1
        idx += 1        
        if count==0:
            break
    
    for i in range(len(new_list)):
        new_list[i] = str(new_list[i])

    answer = ''.join(new_list)
    return answer
    
print(solution(1924,2))
print(solution(1231234,3))
print(solution(4177252841,4))