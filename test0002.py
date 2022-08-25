def solution(number, k):
    stack = []
    for num in number:
        while stack and stack[-1] < num and k > 0:
            if stack[-1]==9:
                pass
            k -= 1
            stack.pop()
        stack.append(num)
    if k != 0:
        stack = stack[:-k]
    return ''.join(stack)

print(solution('1924',2))
print(solution('1231234',3))
print(solution('4177252841',4))
