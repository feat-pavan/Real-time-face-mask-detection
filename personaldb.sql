#use ant one you fell good 
create table personalde(
id int(5) auto_increment,
name char(30) not null,
age int(10) not null,
email varchar(30),
ph_no varchar(20) not null,
gender varchar(30) not null,
address varchar(100),
image longblob not null,
constraint pkb primary key(ph_no))engine='innodb';

create table personalde(
id int auto_increment,
name char(30) not null,
age int8 not null,
email varchar(30) unique not null,
ph_no varchar(20) unique not null,
gender varchar(30) not null,
address varchar(100),
image longblob not null,
constraint pkb primary key(id))engine='innodb';


INSERT INTO `personaldetails` (`name`, `age`, `email`, `ph_no`, `gender`, `address`, `image`) VALUES
('Mohammad Usman Sharif', '23', 'XXXXXXXXXX.com', 'XX7XXXXX3', 'boy', '1130/1 XXXXXXX india', '')

 alter table `personaldetails` modify address varchar(100) not null;   

  alter table `personaldetails` modify image longblob not null;   

INSERT INTO `Users` (`name`, `age`, `email`, `ph_no`, `gender`, `address`) VALUES ('Usman', '23', 'XXXXXXX@gmail.com', 'XXXXXX313', 'Male', ' india karnataka');
