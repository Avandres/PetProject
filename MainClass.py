
# У этого класса должны быть следующие методы:
# 1. Создание базы данных
# 2. Удаление базы данных
# 3. Перезапись данных в базе данных
# 4. Добавление данных в базу данных
# 5. Загрузка данных из базы данных

import sqlite3
import os
import pandas as pd

class MyClass():

    def __init__(self, path_to_database):
        self.connection = sqlite3.connect(path_to_database)

    def create_database(self, filename):
        conn = sqlite3.connect(filename)

    def delete_database(self, filename):
        if filename.endswith('.db'):
            os.remove(filename)
        else:
            raise TypeError('Выбран не тот файл.')

    def rewrite_data_in_database(self, filename, start_id, stop_id):
        pass

    def load_data_from_database(self):
        df = pd.read_sql("SELECT * FROM data", self.connection)
        return df


if __name__ == '__main__':
    MyClass().create_database('test.db')
