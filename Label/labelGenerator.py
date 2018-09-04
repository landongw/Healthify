import sqlite3
import os

conn = sqlite3.connect('label.db')
c = conn.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS LABEL(id REAL, sex TEXT, age REAL, condition TEXT)")

create_table()


os.chdir("D:\PTBDB-DataBase")
directories = os.listdir()
# print(directories)
for folder in directories:
    dir = "D:\PTBDB-DataBase\{}".format(folder)
    if (os.path.isdir(dir)):
        name = int(folder[7:10])
        os.chdir(dir)
        files = os.listdir()
        for file in files:
            if ("Header.txt" in file):
                lines = open(file,'r').readlines()
                for line in lines:
                    if ("# sex:" in line):
                        if ("female" in line):
                            sex = 'female'
                        elif ("male" in line):
                            sex = 'male'
                        else:
                            sex = 'n/a'
                        # print(sex)
                    if ("# age:" in line):
                        age = line[-3:-1]
                        try:
                            age = int(age)
                        except Exception as e:
                            age = -1
                        # print(age)
                    if ("Reason for admission:" in line):
                        pos = line.find(':') + 2
                        condition = line[pos:-1]

        c.execute("INSERT INTO LABEL(id, sex, age, condition) VALUES (?, ?, ?, ?)",
                  (name, sex, age, condition))
        conn.commit()


c.close()
conn.close()
