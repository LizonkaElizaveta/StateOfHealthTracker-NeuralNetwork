import random

afile = open("data/Testing_validation_patients.csv", "w")


def fun_patient():
    line = str(random.uniform(3, 4))  # скорость
    afile.write(line + ',')
    line = str(random.uniform(8.4, 10.6))  # левенштейн
    afile.write(line + ',')
    line = str(random.uniform(90000, 110000))  # время печати в млс
    afile.write(line + ',')
    line = str(random.uniform(10, 15))  # число промахов
    afile.write(line + ',')
    line = str(random.uniform(20000, 3704))  # расстояние от кнопки
    afile.write(line + ',')
    line = str(random.uniform(14, 20))  # таппинг тест правой рук
    afile.write(line + ',')
    line = str(random.uniform(20, 23))  # таппинг текст левой руки
    afile.write(line + ',')
    line = str(random.uniform(55000, 80300))  # расстояние фигуры
    afile.write(line + ',')  # пациенты
    line = str(1)  # состояние
    afile.write(line + '\n')

pass


def fun_health():
    line = str(random.uniform(4, 6.5))  # скорость
    afile.write(line + ',')
    line = str(random.uniform(3, 6))  # левенштейн
    afile.write(line + ',')
    line = str(random.uniform(40000, 70000))  # время печати в млс
    afile.write(line + ',')
    line = str(random.uniform(0,4))  # число промахов
    afile.write(line + ',')
    line = str(random.uniform(9000, 100000))  # расстояние от кнопки
    afile.write(line + ',')
    line = str(random.uniform(40, 50))  # таппинг тест правой рук
    afile.write(line + ',')
    line = str(random.uniform(44, 47))  # таппинг текст левой руки
    afile.write(line + ',')
    line = str(random.uniform(12000, 13000))  # расстояние фигуры
    afile.write(line + ',')
    line = str(0)  # состояние
    afile.write(line + '\n')


pass

for i in range(50000):
    if random.randint(0, 1) == 1:
        fun_patient()
    else:
        fun_health()
afile.close()