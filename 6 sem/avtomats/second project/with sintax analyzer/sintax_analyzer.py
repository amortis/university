import string
import os

ALPHABET = string.ascii_uppercase + string.ascii_lowercase
DIGITS = string.digits
NON_ZERO_DIGITS = DIGITS[1:]


class Analyzer:
    def __init__(self, s: str, file_name: str):
        self.input_string = s
        self.file_name = file_name
        self.pos = 0
        self.x = None
        self.y = None
        self.n1 = None
        self.n2 = None
        self.var_count = 0
        self.num_count = 0

    def error(self, n: int, message: str):
        print(f"Ошибка в символе {n}: {message}")
        return False

    def sintax_error(self, message: str):
        print(f"Синтаксическая ошибка: {message}")
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
        if self.x == self.y:
            return self.sintax_error("Переменные x и y не должны быть одинаковыми")
        if self.n1 != self.n2:
            return self.sintax_error("Числа n должны быть одинаковыми")
        return self.generate_file()

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
        var = self.input_string[start_pos:self.pos]
        self.var_count += 1
        if self.var_count == 1:
            self.x = var
        elif self.var_count == 2:
            self.y = var
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
        num = self.input_string[start_pos:self.pos]
        self.num_count += 1
        if self.num_count == 1:
            self.n1 = num
        elif self.num_count == 2:
            self.n2 = num
        return True

    def generate_file(self):
        if self.x is None or self.y is None or self.n1 is None or self.n2 is None:
            return False
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
        with open(f"{self.file_name}", "w") as f:
            f.write("using System;\n\n"
                    "class Program\n"
                    "{\n"
                    "\tstatic void Main()\n"
                    "\t{\n")

            f.write(f'\t\tConsole.WriteLine("Enter {self.x}: ");\n')
            f.write(f'\t\tint {self.x} = int.Parse(Console.ReadLine());\n')
            f.write(f'\t\tConsole.WriteLine("Enter {self.y}: ");\n')
            f.write(f'\t\tint {self.y} = int.Parse(Console.ReadLine());\n')
            f.write(f"\t\tdouble result = Math.Pow({self.x}, 2 * {self.n1}) + Math.Pow({self.y}, 2 * {self.n1});\n")
            f.write('\t\tConsole.WriteLine($"Result: {result}");\n')
            f.write("\t}\n"
                    "}\n")
        return True



def run_tests():
    tests = [
        ("x^(2*5)+y^(2*5)", True, "Корректное выражение", {"x": "x", "y": "y", "n": "5"}),
        ("a^(2*1)+b^(2*1)", True, "Корректное с минимальными числами", {"x": "a", "y": "b", "n": "1"}),
        ("var123^(2*9)+other456^(2*9)", True, "Корректное с длинными переменными",
         {"x": "var123", "y": "other456", "n": "999"}),
        ("x^(2*5)+x^(2*5)", False, "Одинаковые переменные", {}),
        ("abc^(2*7)+abc^(2*7)", False, "Одинаковые длинные переменные", {}),
        ("x^(2*5)+y^(2*9)", False, "Разные числа n", {}),
        ("var1^(2*123)+var2^(2*456)", False, "Разные числа n с длинными переменными", {}),
        ("x^(2*5)+y^(2*05)", False, "Число начинается с 0", {}),
        ("x^(2*5)y^(2*5)", False, "Отсутствует '+'", {}),
        ("x^(2*5)+^(2*5)", False, "Переменная начинается не с буквы", {}),
        ("1x^(2*5)+y^(2*5)", False, "Переменная начинается с цифры", {}),
        ("x#^(2*5)+y^(2*5)", False, "Недопустимый символ в переменной", {}),
        ("x^2*5)+y^(2*5)", False, "Отсутствует '(' в степени", {}),
        ("x^(3*5)+y^(2*5)", False, "Неверное число вместо '2'", {}),
        ("x^(2-5)+y^(2*5)", False, "Отсутствует '*' в степени", {}),
        ("x^(2*5+y^(2*5)", False, "Отсутствует ')' в степени", {}),
        ("x^(2*5)+y^(2*5)extra", False, "Лишние символы в конце", {}),
        ("", False, "Пустая строка", {}),
        ("x^(2*5)", False, "Неполное выражение", {}),
        ("x^(2*abc)+y^(2*5)", False, "Неверное число в степени", {}),
        ("x123^(2*999)+x123^(2*999)", False, "Одинаковые переменные с цифрами", {}),
        ("z^(2*1)+w^(2*999)", False, "Разные числа n с короткими переменными", {}),
        ("x^(2*0)+y^(2*0)", False, "Число начинается с 0", {}),
    ]

    for i, (input_str, expected, description, expected_values) in enumerate(tests, 1):
        print(f"\nТест {i}: '{input_str}' ({description})")
        analyzer = Analyzer(input_str, f"files/program{i}.cs")
        result = analyzer.f()
        status = "Пройден" if (result == expected or
                               (expected and isinstance(result, tuple) and result[0] == expected_values[
                                   "x"])) else "Провален"
        print(f"Результат: {result}, Ожидалось: {expected}, Статус: {status}")
        if expected and isinstance(result, tuple):
            x, y, n = result
            values_correct = (x == expected_values["x"] and
                              y == expected_values["y"] and
                              n == expected_values["n"])
            print(f"Значения: x='{x}', y='{y}', n='{n}'")
            print(f"Ожидалось: x='{expected_values['x']}', y='{expected_values['y']}', n='{expected_values['n']}'")
            print(f"Значения корректны: {values_correct}")
            if not values_correct:
                status = "Провален (неверные значения)"
                print(f"Статус: {status}")

run_tests()