import random

afile = open("data/Random_patients.csv", "w")


def fun_patient():
    line = str(random.uniform(1, 2))  # скорость
    afile.write(line + ',')
    line = str(random.uniform(8.4, 19.6))  # левенштейн
    afile.write(line + ',')
    line = str(random.uniform(90000.000, 180000.000))  # время печати в млс
    afile.write(line + ',')
    line = str(random.uniform(10, 20))  # число промахов
    afile.write(line + ',')
    line = str(random.uniform(2050, 4704))  # расстояние от кнопки
    afile.write(line + ',')
    line = str(random.uniform(14, 20))  # таппинг тест правой рук
    afile.write(line + ',')
    line = str(random.uniform(20, 23))  # таппинг текст левой руки
    afile.write(line + ',')
    line = str(random.uniform(550, 1030))  # расстояние фигуры
    afile.write(line + ',1\n')  # пациенты


pass


def fun_health():
    line = str(random.uniform(4, 5.5))  # скорость
    afile.write(line + ',')
    line = str(random.uniform(1, 4))  # левенштейн
    afile.write(line + ',')
    line = str(random.uniform(40000.000, 70000.000))  # время печати в млс
    afile.write(line + ',')
    line = str(random.uniform(0, 1))  # число промахов
    afile.write(line + ',')
    line = str(random.uniform(20, 100))  # расстояние от кнопки
    afile.write(line + ',')
    line = str(random.uniform(47, 70))  # таппинг тест правой рук
    afile.write(line + ',')
    line = str(random.uniform(40, 65))  # таппинг текст левой руки
    afile.write(line + ',')
    line = str(random.uniform(5, 10))  # расстояние фигуры
    afile.write(line + ',0\n')


pass

for i in range(560000):
    if random.randint(0, 1) == 1:
        fun_patient()
    else:
        fun_health()
afile.close()