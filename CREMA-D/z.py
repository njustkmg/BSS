import math

lambda_0 = 0.1
T_grow = 40
epoch = 41
lambda_t = min(1, math.sqrt((1 - lambda_0 ** 2) / T_grow * (epoch + 1) + lambda_0 ** 2))

print(lambda_t)