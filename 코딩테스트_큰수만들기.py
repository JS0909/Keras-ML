import numpy as np

def solution(number, k):
    
    nl = list(number)
    # 문자열의 각 요소를 따로 떼어내서 리스트로 만듦
    
    for i in range(len(nl)):
        nl[i] = int(nl[i])
    # 각 요소들을 정수로 바꿈
    
    count = len(nl)-k # 현재 숫자들 개수에서 k개를 뺀 것만큼의 길이로 숫자를 만들거니까
    idx = 0           # 초기 인덱스 값 줌
    new_list = []     # 답이 될 숫자들을 넣을 공간
    while nl:         # nl리스트 내용(길이)만큼 반복할 것
        idx += 1      # 다음번 비교할 범위리스트는 argmax로 나온 인덱스 다음꺼부터 해야되니까 인덱스+1 해줌
        now_list = nl[idx:len(nl)-(count-1)] # 비교할 범위를 정해줄때 원래 숫자리스트에서 지금 count보다 -1개만큼 뒤에 남겨놔야됨, 그래야 남은 숫자를 뽑지
        idx = np.argmax(now_list)+idx        # 쉽게 말하면 1259에서 count가 2라면 처음 돌땐 125에서만 제일 큰 값을 뽑아야 적어도 다음에 뽑을 하나가 남을 수 있음
        new_list.append(nl[idx])             # 지정한 범위에서 제일 큰 곳의 인덱스 값을 반환해주고 해당 자리의 숫자를 답이 될 공간에 저장함                      
        count -= 1                           # 주의점은 첫번째 이후 argmax 인덱스 값은 범위 지정을 위해 새로 만든 리스트의 인덱스 값으로 계산되기때문에
        if count==0:                         # 앞전의 인덱스를 더해야 원래 리스트에서의 인덱스값으로 쓸 수 있음 
            break                            # 수를 하나 뽑았으니 남은 뽑을 개수를 하나 차감함, 이때 뽑을 개수가 0이면 다뽑은거니까 반복 종료
                                             
                                 
    for i in range(len(new_list)):
        new_list[i] = str(new_list[i])

    answer = ''.join(new_list)
    return answer
    
print(solution('1924',2))
print(solution('1231234',3))
print(solution('4177252841',4))
