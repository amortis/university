import random


def max_unique_numbers(arr):
    res = set()

    arr.sort(reverse=True)
    for num in arr:
        while True:
            if num not in res:
                res.add(num)
                break
            if num == 0:
                break
            num //= 2
    return len(res)


def run_tests():
    print("Базовые тесты:")
    print(max_unique_numbers([-4, -44, 2]))  # Ожидается: 3
    print(max_unique_numbers([1, 1, 1, 2, 2]))  # Ожидается: 3

    print("\nДополнительные тесты:")
    print(max_unique_numbers([1, 2, 3, 4, 5]))  # Ожидается: 5
    print(max_unique_numbers([8, 8, 8, 8]))  # Ожидается: 4
    print(max_unique_numbers([1, 1, 1]))  # Ожидается: 2
    print(max_unique_numbers([0, 0, 0]))  # Ожидается: 1
    print(max_unique_numbers([1024] * 10))  # Ожидается: 10
    print(max_unique_numbers([3, 6, 12]))  # Ожидается: 3

run_tests()

#n = int(input()) # количество чисел
nums = [random.randint(-100, 100) for _ in range(100000)]
print(max_unique_numbers(nums))
