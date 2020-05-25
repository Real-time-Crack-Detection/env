# coding: utf-8

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Image(db.Model):
    __tablename__ = 'image'

    path = db.Column(db.String(100))
    save_date = db.Column(db.DateTime)
    code = db.Column(db.String(50))
    comment = db.Column(db.String(100))
    title1 = db.Column(db.String(50))
    title2 = db.Column(db.String(50))

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.String(50), primary_key=True)
    password = db.Column(db.String(50))
    name = db.Column(db.String(50))
    email = db.Column(db.String(50))
    phonenumber = db.Column(db.String(50))
    code = db.Column(db.String(50))