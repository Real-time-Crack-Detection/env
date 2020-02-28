# coding: utf-8
'''
from sqlalchemy import Column, DateTime, MetaData, String, Table, Integer
from sqlalchemy.dialects.mysql import INTEGER

#metadata = MetaData()

t_image = Table(
    'image', metadata,
    Column('number', INTEGER(11)),
    Column('path', String(100)),
    Column('save_date', DateTime)
)

t_user = Table(
    'user', metadata,
    Column('id', String(50)),
    Column('password', String(50))
)
'''

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Image(db.Model):
    __tablename__ = 'image'

    number = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(100))
    save_date = db.Column(db.DateTime)

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.String(50), primary_key=True)
    password = db.Column(db.String(50))
    name = db.Column(db.String(50))
    email = db.Column(db.String(50))
    phonenumber = db.Column(db.String(50))
    code = db.Column(db.String(50))