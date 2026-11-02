import mysql.connector

from config import *

def get_connection():

    return mysql.connector.connect(

        host=MYSQL_HOST,

        user=MYSQL_USER,

        password=MYSQL_PASSWORD,

        database=MYSQL_DATABASE,

        port=MYSQL_PORT
    )