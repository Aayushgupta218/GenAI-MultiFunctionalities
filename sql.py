import sqlite3

connection = sqlite3.connect('data.db')
cursor = connection.cursor()

# cursor.execute("DROP TABLE IF EXISTS data;")
# cursor.execute("SELECT * FROM data;")
# rows = cursor.fetchall()

# for row in rows:
#     print(row)

connection.commit()
connection.close()


# import sqlite3

# connection  = sqlite3.connect('student.db')

# cursor = connection.cursor()

# table_info =  """
# Create table student(NAME VARCHAR(20),
# ROLLNO INTEGER PRIMARY KEY,
# MARKS INTEGER, SECTION VARHCAR(25));
# """

# cursor.execute(table_info)

# cursor.execute("INSERT INTO student VALUES('John', 101, 90, 'A')")
# cursor.execute("INSERT INTO student VALUES('shaun', 102, 67, 'B')")
# cursor.execute("INSERT INTO student VALUES('kane', 103, 70, 'A')")
# cursor.execute("INSERT INTO student VALUES('Jonny', 104, 95, 'C')")

# print("Data inserted successfully")

# data = cursor.execute("SELECT * FROM student")

# for row in data:
#     print(row)

# connection.commit()
# connection.close()
