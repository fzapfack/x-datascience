--
-- 	Database Table Creation
--
--		This file will create the tables for use with the book 
--  Database Management Systems by Raghu Ramakrishnan and Johannes Gehrke.
--  It is run automatically by the installation script.
--
--	Version 0.1.0.0 2002/04/05 by: David Warden.
--	Copyright (C) 2002 McGraw-Hill Companies Inc. All Rights Reserved.
--
--  First drop any existing tables. Any errors are ignored.
--
drop table student cascade constraints;
drop table faculty cascade constraints;
drop table class cascade constraints;
drop table enrolled cascade constraints;
--
-- Now, add each table.
--
create table Sailors1(
	sid integer,
	sname char(30),
	rating integer,
	age real,
	primary key sid
	);
create table Sailors2(
        sid integer,
        sname char(30),
        rating integer,
        age real,
	primary key sid;
        );
create table Boats(
	bid integer primary key,
	color char(20)
	);
create table Reserves(
	sid integer,
	bid integer,
	day date,
	primary key(sid,bid),
	foreign key(sid) references sailors1(sid),
	foreign key(bid) references boats(sid)
	);
--
-- Exit the Script.
--
quit;
