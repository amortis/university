import string
ALPHABET = string.ascii_uppercase + string.ascii_lowercase
DIGITS = string.digits
NON_ZERO_DIGITS = DIGITS[1:]


class Analyzer:
    def __init__(self, s: str):
        self.input_string = s
        self.pos = 0

    def error(self, n: int, message: str):
        print(f"Ошибка в символе {n}: {message}")
        return False

    def f(self):
        result = self.addent()
        if not result:
            return False
        if self.pos >= len(self.input_string) or self.input_string[self.pos] != '+':
            return self.error(self.pos if self.pos < len(self.input_string) else len(self.input_string) - 1, "Ожидается '+'")
        self.pos += 1
        result = self.addent()
        if not result:
            return False
        if self.pos < len(self.input_string):
            return self.error(self.pos, "Лишние символы после выражения")
        return True

    def addent(self):
        start_pos = self.pos
        if not self.value():
            return False
        if self.pos >= len(self.input_string) or self.input_string[self.pos] != '^':
            return self.error(self.pos if self.pos < len(self.input_string) else start_pos, "Ожидается '^'")
        self.pos += 1
        if not self.power():
            return False
        return True

    def value(self):
        start_pos = self.pos
        if self.pos >= len(self.input_string) or self.input_string[self.pos] not in ALPHABET:
            return self.error(self.pos if self.pos < len(self.input_string) else start_pos, "Ожидается переменная (буква)")
        self.pos += 1
        while self.pos < len(self.input_string) and (self.input_string[self.pos] in ALPHABET or self.input_string[self.pos] in DIGITS):
            self.pos += 1
        return True

    def power(self):
        start_pos = self.pos
        if self.pos >= len(self.input_string) or self.input_string[self.pos] != '(':
            return self.error(self.pos if self.pos < len(self.input_string) else start_pos, "Ожидается '('")
        self.pos += 1
        if self.pos >= len(self.input_string) or self.input_string[self.pos] != '2':
            return self.error(self.pos if self.pos < len(self.input_string) else start_pos, "Ожидается '2'")
        self.pos += 1
        if self.pos >= len(self.input_string) or self.input_string[self.pos] != '*':
            return self.error(self.pos if self.pos < len(self.input_string) else start_pos, "Ожидается '*'")
        self.pos += 1
        if not self.number():
            return False
        if self.pos >= len(self.input_string) or self.input_string[self.pos] != ')':
            return self.error(self.pos if self.pos < len(self.input_string) else start_pos, "Ожидается ')'")
        self.pos += 1
        return True

    def number(self):
        start_pos = self.pos
        if self.pos >= len(self.input_string) or self.input_string[self.pos] not in NON_ZERO_DIGITS:
            return self.error(self.pos if self.pos < len(self.input_string) else start_pos, "Ожидается число, начинающееся с 1-9")
        self.pos += 1
        while self.pos < len(self.input_string) and self.input_string[self.pos] in DIGITS:
            self.pos += 1
        return True


def run_tests():
    tests = [
        ("x^(2*5)+y^(2*9)", True, "Корректное выражение"),
        ("a^(2*1)+b^(2*123)", True, "Корректное выражение с минимальными числами"),
        ("longVar123^(2*999)+anotherVar456^(2*1)", True, "Корректное выражение с длинными переменными"),
        ("x^(2*5)y^(2*9)", False, "Отсутствует '+'"),
        ("x^(2*5)+^(2*9)", False, "Переменная начинается не с буквы"),
        ("1x^(2*5)+y^(2*9)", False, "Переменная начинается с цифры"),
        ("x#^(2*5)+y^(2*9)", False, "Недопустимый символ в переменной"),
        ("x^2*5)+y^(2*9)", False, "Отсутствует '(' в степени"),
        ("x^(3*5)+y^(2*9)", False, "Неверное число вместо '2'"),
        ("x^(2-5)+y^(2*9)", False, "Отсутствует '*' в степени"),
        ("x^(2*0)+y^(2*9)", False, "Число начинается с 0"),
        ("x^(2*5+y^(2*9)", False, "Отсутствует ')' в степени"),
        ("x^(2*5)+y^(2*9)extra", False, "Лишние символы в конце"),
        ("", False, "Пустая строка"),
        ("x^(2*5)", False, "Неполное выражение, отсутствует второе слагаемое"),
        ("x^(2*abc)+y^(2*9)", False, "Неверное число в степени"),
    ]

    print("Запуск тестов...")
    for i, (input_str, expected, description) in enumerate(tests, 1):
        print(f"\nТест {i}: '{input_str}' ({description})")
        analyzer = Analyzer(input_str)
        result = analyzer.f()
        status = "Пройден" if result == expected else "Провален"
        print(f"Результат: {result}, Ожидалось: {expected}, Статус: {status}")

run_tests()